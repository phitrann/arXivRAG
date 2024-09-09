# Makefile for dev stage

.PHONY: all setup minio milvus ollama mongodb

# Define the Docker network name
DOCKER_NETWORK = arxivrag_network

# Setup the environment
all: setup ollama minio milvus
vectordb: setup minio milvus
llm: setup ollama

# Create Docker network
setup:
	@docker network inspect $(DOCKER_NETWORK) >/dev/null 2>&1 || docker network create $(DOCKER_NETWORK)
	@echo "Docker network $(DOCKER_NETWORK) created or already exists."

# Run ollama setup
ollama:
	@mkdir -p ollama/.data
	@docker stack deploy -c ollama/ollama-docker-compose.yml ollama
	@echo "Ollama services are up and running."

# Run MinIO setup
minio:
	@mkdir -p minio/.data
	@docker compose -f minio/minio-docker-compose.yml up -d
	@echo "MinIO services are up and running."

# Run Milvus setup
milvus:
	@mkdir -p milvus/.data/etcd milvus/.data/milvus
	@docker compose -f milvus/milvus-docker-compose.yml up -d
	@echo "Milvus services are up and running."

# Run MongoDB setup
mongodb:
	@mkdir -p mongodb/.data mongodb/.data/mongodb_storage mongodb/.data/mongodb_configdb
	@docker compose -f mongodb/mongo-docker-compose.yml up -d
	@echo "MongoDB services are up and running."

