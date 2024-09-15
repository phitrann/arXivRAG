# arXivRAG
![](https://geps.dev/progress/30)

## Overview

**arXivRAG** is a comprehensive tool designed to enhance the retrieval and generation of academic content from the arXiv database. Leveraging advanced Retrieval-Augmented Generation (RAG) techniques, arXivRAG provides researchers, students, and enthusiasts with the ability to discover and generate summaries, insights, and analyses of arXiv papers efficiently.

## 🔍 Features
### Core features
- **Retrieval-Augmented Generation**: Combines the power of retrieval systems with generative models to enhance the accuracy and relevance of responses.
- **arXiv Integration**: Directly queries the arXiv repository to fetch and summarize academic papers.
- **User-Friendly Interface**: Provides an easy-to-use interface for querying and obtaining summaries of scientific papers.
- **Customizable**: Allows users to customize the retrieval and generation parameters to suit their specific needs.

### Advance features
- [ ] **Enhanced Search**: Advanced search capabilities to quickly find relevant papers.
- [ ] **Summarization**: Automatic generation of concise summaries for arXiv papers.
- [ ] **Custom Queries**: Tailored query support to retrieve specific information from academic papers.
- [ ] **Real-Time Access**: Seamless integration with the arXiv API for real-time data access.
- [ ] **Citation and Trend Analysis**: Analyze citation networks, visualize the impact of papers, and identify emerging research trends based on recent publications and citation patterns.

## 🚀 Installation

To get started with arXivRAG, follow these steps:

1. Clone the repository:
```
git clone https://github.com/phitrann/arXivRAG.git
cd arXivRAG
```

2. Create a virtual environment (we recommend using conda):
```
conda create -n arxiv-rag python=3.10
conda activate arxiv-rag
```

3. Install the required dependencies:
```
pip install -r requirements.txt
```

## 💻 Usage

To use arXivRAG, follow these steps:

1. Run the main script:
```
python main.py
```

3. Query the system:
- Enter your query related to a scientific paper.
- The system will retrieve relevant papers from arXiv and generate a summary.

## Configuration
You can customize the behavior of arXivRAG by modifying the configuration file `config.yaml`. Key parameters include:

- **retrieval_model**: The model used for retrieving relevant papers.
- **generation_model**: The model used for generating summaries.
- **num_retrievals**: The number of papers to retrieve for each query.
- **max_summary_length**: The maximum length of the generated summary.

## ❤️ Contributing
[![Contributors](https://contrib.rocks/image?repo=phitrann/arXivRAG&max=10)](https://github.com/phitrann/arXivRAG/graphs/contributors)

We welcome contributions from the community! If you have ideas for new features or improvements, feel free to open an issue or submit a pull request. 

In case you want to submit a pull request, please follow these steps:

1. Fork the repository.
2. Create a new branch:
```
git checkout -b feature/your-feature-name
```

4. Make your changes and commit them:
```
git commit -m "Add your commit message"
```

5. Push to the branch:
```
git push origin feature/your-feature-name
```

6. Create a pull request.

## 📜 License

This project is released under the [Apache 2.0 license](https://github.com/phitrann/arXivRAG/blob/main/LICENSE). See the LICENSE file for details.

## Acknowledgements
- Thanks to the contributors of the arXivRAG project.
- Special thanks to the developers of the retrieval and generation models used in this project.
