## Usage

### Prepare the .env file that 
```bash
# .env file
MILVUS_URI=
OLLAMA_BASE_URL=
EMBEDDING_DIM=
```

### Prepare the .csv file that have the column abstract
| abstract                                                                                                          | authors                                                                                     | n_citation | references                                                                                               | title                                                      | venue                                      | year | id                                      |
|-------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|------------|----------------------------------------------------------------------------------------------------------|------------------------------------------------------------|-------------------------------------------|------|------------------------------------------|
| In this paper, a robust 3D triangular mesh wat...                                                                 | ['S. Ben Jabra', 'Ezzeddine Zagrouba']                                                      | 50         | ['09cb2d7d-47d1-4a85-bfe5-faa8221e644b', '10aa...                                                   | A new approach of 3D watermarking based on ima...          | international symposium on computers and commu... | 2008 | 4ab3735c-80f1-472d-b953-fa0557fed28b     |
| We studied an autoassociative neural network w...                                                                 | ['Joaquín J. Torres', 'Jesús M. Cortés', 'Joaq...                                            | 50         | ['4017c9d2-9845-4ad2-ad5b-ba65523727c5', 'b118...                                                   | Attractor neural networks with activity-depend...          | Neurocomputing                            | 2007 | 4ab39729-af77-46f7-a662-16984fb9c1db     |
| It is well-known that Sturmian sequences are t...                                                                | ['Genevi eve Paquin', 'Laurent Vuillon']                                                    | 50         | ['1c655ee2-067d-4bc4-b8cc-bc779e9a7f10', '2e4e...                                                   | A characterization of balanced episturmian seq...         | Electronic Journal of Combinatorics        | 2007 | 4ab3a4cf-1d96-4ce5-ab6f-b3e19fc260de     |

### 1. Run the question generation
```bash
python question_generation.py
```
or in my example
```bash
python question_generation.py --csv_file_path './dataset/dblp-v10.csv' --sample_size 100 --model_generate 'phi3' --model_embedding 'BAAI/bge-small-en-v1.5'
```

### 2. Run the evaluation 

#### For sparse or hybrid

```bash
python retriever_evaluation.py 
```
or in my example
```bash
python retriever_evaluation.py --data_dir "./data" \
                           --chunk_size 512 \
                           --top_k 2 \
                           --qa_file "./output_evaluation/pg_eval_dataset.json" \
                           --mode hybrid \
                           --llm_model_name "phi3" \
                           --embedding_model_name "BAAI/bge-small-en-v1.5"
```

#### For dense evaluation

```bash 
python retriever_dense.py
```


