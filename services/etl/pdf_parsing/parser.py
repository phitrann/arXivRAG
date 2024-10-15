import pathlib
import sys
import time
import os
import fitz  # PyMuPDF
import io
from datetime import datetime, timedelta
from minio import Minio
from loguru import logger

from .pdf2md import to_markdown
from .utils import load_models_and_params

class PDFParser:
    def __init__(self, minio_client: Minio = None, bucket_name: str = "arxiv-papers", parms=None):
        """
        Initializes the PDFParser with optional MinIO client support.

        Args:
            minio_client (Minio): Optional MinIO client for handling remote files.
            bucket_name (str): MinIO bucket name for remote file storage.
            parms (list): List of parameters for page selection, format like ['-pages', '1-3,5,N-2'].
        """
        self.doc = None
        self.pages = None
        self.md_string = None
        
        self.mfd_model, self.layout_model, self.img_size, self.mfd_conf_thres, self.mfd_iou_thres = load_models_and_params()
        
        self.minio_client = minio_client
        self.bucket_name = bucket_name

        self.base_pdf_path = "papers"
        self.base_processed_path = "processed_papers"
        self.base_figures_path = "figures"
        
        # Initialize parms (can be passed as an argument)
        self.parms = parms if parms else []

    def open_document(self, file_data=None, file_path=None):
        """
        Opens a PDF document either from file data (MinIO object) or from a file path.
        
        Args:
            file_data (bytes): Binary data of the PDF file from MinIO.
            file_path (str): Path to the PDF file for local files.
        """
        if file_data:
            self.doc = fitz.open("pdf", file_data)
        elif file_path:
            self.doc = fitz.open(file_path)

    def set_pages(self):
        """Sets the range of pages to be parsed from the document."""
        self.pages = range(self.doc.page_count)  # default page range
        if len(self.parms) == 2 and self.parms[0] == "-pages":  # page sub-selection given
            self.pages = []  # list of desired page numbers

            # replace any variable "N" by page count
            pages_spec = self.parms[1].replace("N", f"{self.doc.page_count}")
            for spec in pages_spec.split(","):
                if "-" in spec:
                    start, end = map(int, spec.split("-"))
                    self.pages.extend(range(start - 1, end))
                else:
                    self.pages.append(int(spec) - 1)

            # make a set of invalid page numbers
            wrong_pages = set([n + 1 for n in self.pages if n >= self.doc.page_count][:4])
            if wrong_pages:  # if any invalid numbers given, exit.
                sys.exit(f"Page number(s) {wrong_pages} not in '{self.doc}'.")

    def to_markdown(self):
        """Converts the PDF to a Markdown format using the loaded models."""
        self.md_string = to_markdown(
            self.doc, 
            pages=self.pages, 
            mfd_model=self.mfd_model, 
            layout_model=self.layout_model, 
            mfd_conf_thres=self.mfd_conf_thres, 
            mfd_iou_thres=self.mfd_iou_thres, 
            img_size=self.img_size
        )

    def save_markdown(self, outname=None):
        """Saves the Markdown output locally."""
        if not outname:
            outname = self.doc.name.replace(".pdf", ".md")
        pathlib.Path(outname).write_bytes(self.md_string.encode())

    def parse_pdf(self, file_path=None, file_data=None, save_to_minio=False, date=None):
        """
        Parses a PDF to Markdown, handling both local files and MinIO objects.
        
        Args:
            file_path (str): The local path of the PDF file.
            file_data (bytes): PDF data as binary (from MinIO).
            save_to_minio (bool): Whether to save the markdown to MinIO.
            date (str): Date string for MinIO path organization.
        """
        t0 = time.perf_counter()  # start a timer
        
        if file_data:
            self.open_document(file_data=file_data)
        elif file_path:
            self.open_document(file_path=file_path)
        else:
            raise ValueError("Either file_path or file_data must be provided.")

        self.set_pages()
        md_string = to_markdown(
                    self.doc, 
                    pages=self.pages, 
                    mfd_model=self.mfd_model, 
                    layout_model=self.layout_model, 
                    mfd_conf_thres=self.mfd_conf_thres, 
                    mfd_iou_thres=self.mfd_iou_thres, 
                    img_size=self.img_size
                )
        # if save_to_minio and self.minio_client:
        #     self.save_markdown_to_minio(date, file_path)
        # else:
        #     self.save_markdown()

        t1 = time.perf_counter()  # stop timer
        print(f"Markdown creation time for {self.doc.name=} {round(t1-t0, 2)} sec.")
        return md_string

    def parse_bulk_pdfs(self, start_date: str, end_date: str, save_to_minio: bool = False):
        """
        Parses PDF files from MinIO bucket within a date range and converts them to Markdown.

        Args:
            start_date (str): Start date in the format 'YYYYMMDD'.
            end_date (str): End date in the format 'YYYYMMDD'.
            save_to_minio (bool): Whether to save the markdown files to MinIO.

        Returns:
            list: A list of dictionaries containing file path, processing status, md_string, and any error message.
        """
        if not self.minio_client:
            raise ValueError("MinIO client is not initialized")

        # Convert date strings to datetime objects
        start_date = datetime.strptime(start_date, "%Y%m%d")
        end_date = datetime.strptime(end_date, "%Y%m%d")

        results = []

        crawler = start_date
        while crawler <= end_date:
            date = crawler.strftime("%Y%m%d")
            folder_path = f"{self.base_pdf_path}/{date}/"

            logger.info(f"Fetching papers for date: {date}")

            # List objects in the MinIO bucket
            paper_objs = self.minio_client.list_objects(self.bucket_name, prefix=folder_path)
            for obj in paper_objs:
                file_path = obj.object_name
                logger.info(f"Starting processing for file: {file_path}")

                try:
                    # Start timing for processing
                    start_time = time.perf_counter()

                    # Get the PDF from MinIO and read data
                    paper = self.minio_client.get_object(self.bucket_name, file_path)
                    paper_data = paper.read()  # Read file bytes

                    # Parse the PDF
                    md_string = self.parse_pdf(file_data=paper_data, save_to_minio=save_to_minio, date=date)

                    # Log success with processing time
                    elapsed_time = time.perf_counter() - start_time
                    logger.info(f"Successfully processed {file_path} in {elapsed_time:.2f} seconds.")

                    # Append result with md_string
                    results.append({
                        'file_path': file_path,
                        'status': 'success',
                        'md_string': md_string,
                        'error': None,
                        'processing_time': f"{elapsed_time:.2f} seconds"
                    })

                except Exception as e:
                    # Log error with file path
                    logger.error(f"Error processing {file_path}: {e}")

                    # Append failure result
                    results.append({
                        'file_path': file_path,
                        'status': 'failed',
                        'md_string': None,
                        'error': str(e),
                        'processing_time': None
                    })

            # Move to the next day
            crawler += timedelta(days=1)

        # Return the results list containing file path, status, and md_string
        return results

            
    def save_markdown_to_minio(self, date, file_path):
        """Saves the generated Markdown file to MinIO."""
        md_byte = io.BytesIO(self.md_string.encode()) 
        md_file_path = f"{self.base_processed_path}/{date}/{file_path.split('/')[-1].replace('.pdf', '.md')}"
        self.minio_client.put_object(
            self.bucket_name,
            md_file_path,
            data=md_byte,
            length=len(md_byte.getvalue()),
            content_type="text/markdown",
        )

if __name__ == "__main__":
    os.environ["NO_PROXY"] = "localhost"

    # Example usage for local PDF
    parser = PDFParser(parms=['-pages', '1-3,5'])
    parser.parse_pdf(file_path="example.pdf")

    # Example usage for MinIO stored PDFs
    client = Minio(
        endpoint="localhost:9800",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False,
    )
    logger.add("logs/pdf_parsing.log", rotation="10 MB")
    logger.info("MinIO client is initialized")

    minio_parser = PDFParser(client)
    minio_parser.parse_pdf_to_markdown_from_minio("20240122", "20240123", save_to_minio=True)
