import scipdf
import os
import json
import io
from datetime import datetime, timedelta
from minio import Minio
from loguru import logger

os.environ["NO_PROXY"] = "localhost"


class ArxivPDFParser:
    def __init__(self, minio_client: Minio, bucket_name: str = "arxiv-papers"):
        self.minio_client = minio_client
        self.bucket_name = bucket_name

        self.base_pdf_path = "papers"
        self.base_processed_path = "processed_papers"
        self.base_figures_path = "figures"

    def parse_pdf_to_dict(self, start_date: str, end_date: str):
        # Parse string to datetime
        start_date = datetime.strptime(start_date, "%Y%m%d")
        end_date = datetime.strptime(end_date, "%Y%m%d")

        crawler = start_date
        # Generate dates from the start to end date
        while crawler <= end_date:
            date = crawler.strftime("%Y%m%d")

            # Get the pdf of the date
            folder_path = f"{self.base_pdf_path}/{date}/"

            logger.info(f"Fetching papers for {date}")

            paper_objs = self.minio_client.list_objects(
                self.bucket_name, prefix=folder_path
            )
            for obj in paper_objs:
                file_path = obj.object_name
                logger.info(f"Processing {file_path}")

                paper = self.minio_client.get_object(self.bucket_name, file_path)

                try:
                    paper_dict = scipdf.parse_pdf_to_dict(paper.data)
                except Exception as e:
                    logger.error(f"Error in parsing PDF '{file_path}' with error: {e}")
                    continue
                paper_dict_byte = io.BytesIO(json.dumps(paper_dict).encode()) 

                # Save the paper_dict to MinIO
                json_file_path = f"{self.base_processed_path}/{date}/{file_path.split('/')[-1].replace('.pdf', '.json')}"
                self.minio_client.put_object(
                    self.bucket_name,
                    json_file_path,
                    data=paper_dict_byte,
                    length=len(paper_dict_byte.getvalue()),
                    content_type="application/json",
                )

                # json_file_path = f"{file_path.split('/')[-1].replace('.pdf', '.json')}"
                # json.dump(paper_dict, open(json_file_path, 'w'))

            crawler += timedelta(days=1)

        # self.article_dict = scipdf.parse_pdf_to_dict(self.file_path)
        # return self.article_dict

    def parse_figures(self, output_folder="figures"):
        self.figures = scipdf.parse_figures(self.file_path, output_folder=output_folder)
        return


if __name__ == "__main__":
    client = Minio(
        endpoint="localhost:9800",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False,
    )
    logger.add("logs/pdf_parsing.log", rotation="10 MB")
    logger.info("MinIO client is initialized")

    parser = ArxivPDFParser(client)
    parser.parse_pdf_to_dict("20240122", "20240821")
