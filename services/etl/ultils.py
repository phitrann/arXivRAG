import io
import json
import os
import requests
import urllib
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Tuple

from minio import Minio

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.milvus import MilvusVectorStore


class Fetcher():
    """Fetcher class to fetch arXiv papers."""

    def __init__(self, categories_list: List[str], arxiv_category_file_path) -> None:
        self.categories_list = categories_list
        self.subcategory_symbol_list = []
        self.symbol_to_name = {}

        # Create subcategories symbol list and mapping
        with open(arxiv_category_file_path, 'r') as file:
            data = json.load(file)

            for category in categories_list:
                for subcategory in data[category].values():
                    self.subcategory_symbol_list.append(subcategory["symbol"])

            for category, subcategories in data.items():
                for name, subcategory in subcategories.items():
                    self.symbol_to_name[subcategory["symbol"]] = name

    def fetch_papers(
            self,
            start_day: int, start_month: int, start_year: int,
            MinIO_client: Minio, bucket_name: str = "arxiv-papers",
            max_results: int = 800
    ) -> Tuple[List[str], List[str]]:
        """
        Fetch arXiv papers based on the chosen categories and start date.

        Args:
            start_day (int): The day to start fetching papers.
            start_month (int): The month to start fetching papers.
            start_year (int): The year to start fetching papers.
            MinIO_client (Minio): The MinIO client to interact with the object storage.
            bucket_name (str): The name of the bucket to store the papers.
            max_results (int): The maximum number of papers to fetch for each day.

        Returns:
            Tuple[List[str], List[str]]: A tuple of lists containing the paths and codes of the fetched papers.
        """

        start_date = datetime(start_year, start_month, start_day, 0, 0)
        current_date = datetime.now()

        # List of formatted dates to fetch arXiv papers
        formatted_dates = []

        # Generate dates from the start of the chosen year to the current date
        while start_date <= current_date:
            formatted_date = start_date.strftime('%Y%m%d*')
            formatted_dates.append(formatted_date)
            start_date += timedelta(days=1)

        # Fetch arXiv papers
        search_query = '+OR+'.join(['cat:' + symbol for symbol in self.subcategory_symbol_list])
        list_of_papers_path = []
        list_of_papers_code = []

        # Generate search queries for each day
        for i in range(len(formatted_dates) - 1):
            search_start_time = formatted_dates[i]
            search_end_time = formatted_dates[i + 1]
            url = f'https://dailyarxiv.com/query.php?search_query={search_query}+AND+lastUpdatedDate:[{search_start_time}+TO+{search_end_time}]&max_results={max_results}'
            data = urllib.request.urlopen(url)
            xml_data = data.read().decode('utf-8')
            # print(xml_data)

            # Parse XML
            root = ET.fromstring(xml_data)
            namespace = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}

            # Count number of papers
            num_papers = len(root.findall('atom:entry', namespace))
            print(f"\nFound {num_papers} papers for the date: {search_start_time.replace('*', '')}")

            for entry in root.findall('atom:entry', namespace):
                title = entry.find('atom:title', namespace).text
                summary = entry.find('atom:summary', namespace).text
                authors = [author.find('atom:name', namespace).text for author in entry.findall('atom:author', namespace)]
                categories = [self.symbol_to_name[category.attrib['term']] for category in entry.findall('atom:category', namespace) if category.attrib['term'] in self.symbol_to_name]
                url = entry.find('atom:link[@rel="alternate"][@type="text/html"]', namespace).attrib['href']
                paper_code = url.split('/')[-1]

                metadata = {
                    "title": title,
                    "abstract": summary.strip(),
                    "authors": authors,
                    "categories": categories,
                    "url": url
                }

                # Create subdirectories based on the date
                date_folder = search_start_time.replace('*', '')
                metadata_path = f'metadata/{date_folder}/{paper_code}.json'
                papers_path = f'papers/{date_folder}/{paper_code}.pdf'

                # Save metadata into MinIO
                metadata_bytes = json.dumps(metadata, indent=4).encode('utf-8')
                MinIO_client.put_object(
                    bucket_name,
                    metadata_path,
                    data=io.BytesIO(metadata_bytes),
                    length=len(metadata_bytes),
                    content_type='application/json'
                )

                # Download PDF
                url = f'https://arxiv.org/pdf/{paper_code}'
                response = requests.get(url)

                # Save PDF
                if response.status_code != 200:
                    print(f"Failed to download PDF file of the paper: {paper_code}")
                    continue
                MinIO_client.put_object(
                    bucket_name,
                    papers_path,
                    data=io.BytesIO(response.content),
                    length=len(response.content),
                    content_type='application/pdf'
                )

                print(f"Saved metadata and downloaded PDF file of the paper: {paper_code}")

                list_of_papers_path.append(papers_path)
                list_of_papers_code.append(paper_code)

        return list_of_papers_path, list_of_papers_code


class ETL():
    """ETL class to extract arXiv papers, transform them into text nodes as embeddings, and load them into the vector store."""

    def __init__(self, root_folder: str, vector_store: MilvusVectorStore, embed_model: OllamaEmbedding, MinIO_client: Minio, bucket_name: str = "arxiv-papers") -> None:
        self.root_folder = root_folder
        self.vector_store = vector_store
        self.embed_model = embed_model
        self.MinIO_client = MinIO_client
        self.bucket_name = bucket_name

    def extract_transform_load(
            self,
            list_of_papers_path: List[str],
            list_of_papers_code: List[str],
            text_parser: SentenceSplitter,
            pdf_reader: PyMuPDFReader
    ) -> None:
        """
        Extract, transform, and load arXiv papers into the vector store.
        
        Args:
            list_of_papers_path (List[str]): A list of paths to the arXiv papers.
            list_of_papers_code (List[str]): A list of codes of the arXiv papers.
            text_parser (SentenceSplitter): The text parser to split text into chunks.
            pdf_reader (PyMuPDFReader): The PDF reader to extract text from PDF.
        
        Returns:
            None
        """

        for paper_path, paper_code in zip(list_of_papers_path, list_of_papers_code):
            # Extract text from PDF
            pdf_data = self.MinIO_client.get_object(self.bucket_name, paper_path)
            pdf_file_path = os.path.join(self.root_folder, f'{paper_code}.pdf')
            with open(pdf_file_path, 'wb') as file_data:
                # Read data from response by each 32KB
                for d in pdf_data.stream(32*1024):
                    file_data.write(d)
            documents = pdf_reader.load(file_path=pdf_file_path)

            # Split documents into chunks
            text_chunks = []
            doc_idxs = []
            for doc_idx, doc in enumerate(documents):
                cur_text_chunks = text_parser.split_text(doc.text)
                text_chunks.extend(cur_text_chunks)
                doc_idxs.extend([doc_idx] * len(cur_text_chunks))

            # Construct text nodes from text chunks
            nodes = []
            for idx, text_chunk in enumerate(text_chunks):
                node = TextNode(text=text_chunk)
                src_doc = documents[doc_idxs[idx]]
                node.metadata = src_doc.metadata
                nodes.append(node)

            # Generate embeddings for text nodes
            for node in nodes:
                node_embedding = self.embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
                node.embedding = node_embedding

            # Load text nodes into the vector store
            self.vector_store.add(nodes)
            print(f"Parsed and indexed PDF file of the paper: {paper_code}")

            # Remove PDF file
            os.remove(pdf_file_path)

        print("\nFinished extracting, transforming, and loading arXiv papers into the vector store.")
