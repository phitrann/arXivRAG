# Welcome to ArXiv papers fetching and ETL process! 🚀🤖

Hi there, Developer! 👋 The following steps will help you perfrom papers fetching and ETL:
- First, you need to run [category_crawler.py](./category_crawler.py) to get list of subcategories ID based on your desire categories, which is stored in [arxiv_category.json](./arxiv_category.json).
- Then run `jobs.py` script to fetch ArXiv papers and perform ETL to transform these papers into important embeddings and load them into vector database. 
    - Sample command to fetch all pappers from Aug 19, 2024 to the current date:
        ```python
        python services/etl/jobs.py --start_day 19 --start_month 8 --start_year 2024
        ```
    - The default embedding model is [BAAI/llm-embedder](https://huggingface.co/BAAI/llm-embedder), which will generate embedding with size `768`. You can modify the model name on your own choice.
    - You also need to fill in the values of needed variables in `.env` file saved in folder [etl](../etl/).
        ```bash
        MINIO_ENDPOINT=
        MINIO_ACCESS_KEY=
        MINIO_SECRET_KEY=
        MINIO_BUCKET_NAME=

        MILVUS_URI=
        MILVUS_COLLECTION_NAME=
        ```
    - By default, the system will use `cuda` as device to run but if you don't have gpu, you can use `cpu` instead, which you need to declare in `--device` when running the script.