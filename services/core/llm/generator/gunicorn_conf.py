# Worker Options
workers = 2
worker_class = "uvicorn.workers.UvicornWorker"
worker_tmp_dir = "/dev/shm" # Use a RAM disk for the worker temporary files

# Address and Port for the workers to bind to
bind = "0.0.0.0:8001"

# Worker timeout
timeout = 120

# Log level
loglevel = "info"

# Preload the app before forking the workers (improves performance)
preload_app = True

# Access log
# accesslog = "./logs/access.log"
# errorlog = "./logs/error.log"