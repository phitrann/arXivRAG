import io
import json
import urllib
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Tuple

import requests
from requests.exceptions import ChunkedEncodingError

from minio import Minio

class DataFetcher():
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
            start_date: str, end_date: str,
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

        start_date = datetime.strptime(start_date, "%Y%m%d")
        end_date = datetime.strptime(end_date, "%Y%m%d")


        # List of formatted dates to fetch arXiv papers
        formatted_dates = []

        # Generate dates from the start of the chosen year to the current date
        while start_date <= end_date:
            formatted_date = start_date.strftime('%Y%m%d')
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
            date = datetime.strptime(search_start_time.replace('*', ''), "%Y%m%d").strftime("%d/%m/%Y")
            print(f"\nFound {num_papers} papers for the date: {date}") 

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
                retries = 3
                backoff_factor = 0.3 # Time (seconds) to wait between retries
                url = f'https://arxiv.org/pdf/{paper_code}'
                for attempt in range(retries):
                    try:
                        response = requests.get(url)
                    except ChunkedEncodingError as e:
                        if attempt < retries - 1:
                            sleep_time = backoff_factor * (2 ** attempt)
                            print(f"ChunkedEncodingError encountered. Retrying in {sleep_time} seconds...")
                            time.sleep(sleep_time)
                        else:
                            print("Max retries reached. Failed to download PDF.")
                            continue

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