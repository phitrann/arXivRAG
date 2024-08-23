from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import json

# Set up the Chrome options
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Run Chrome in headless mode
options.add_argument('--disable-gpu')  # Disable GPU acceleration
options.add_argument('--no-sandbox')  # Bypass OS security
options.add_argument('--incognito')  # Enable incognito mode
options.add_argument('--ignore-certificate-errors')  # Ignore certificate errors
options.add_argument('--ignore-ssl-errors')  # Ignore SSL errors
options.add_argument('--disable-dev-shm-usage')  # Disable shared memory usage
driver = webdriver.Chrome(options=options)  # Use Chrome webdriver

# Set up the Chrome driver
driver = webdriver.Chrome(options=options)

# Open the URL
URL = 'https://arxiv.org/category_taxonomy'
driver.get(URL)
time.sleep(5)

# Load the entire page
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
page_source = driver.page_source

# Parse the HTML content
page = BeautifulSoup(page_source, 'html.parser')

# Find element containing category list
category_list = page.find('div', {'id': 'category_taxonomy_list'})
category_dict = {}
for category in category_list.find_all('h2', {'class': 'accordion-head'}):
    category_dict[category.text] = {}

for idx, category_item in enumerate(category_list.find_all('div', {'class': 'accordion-body'})):
    sub_category_dict = {}
    for sub_category_item in category_item.find_all('div', {'class': 'columns divided'}):
        try:
            sub_category_symbol = sub_category_item.find('div', {'class': 'column is-one-fifth'}).find('h4').contents[0].strip()
            sub_category_name = sub_category_item.find('div', {'class': 'column is-one-fifth'}).find('h4').find('span').text.strip('()')
            sub_category_description = sub_category_item.find_all('div', {'class': 'column'})[1].text.strip()

            sub_category_dict[sub_category_name] = {
                'symbol': sub_category_symbol,
                'description': sub_category_description
            }
        except AttributeError as e:
            print(f"Error processing sub category item: {e}")

    category_dict[list(category_dict.keys())[idx]] = sub_category_dict

# Convert to JSON
with open('arxiv_category.json', 'w') as f:
    json.dump(category_dict, f, indent=4)

# Close the webdriver
driver.quit()
