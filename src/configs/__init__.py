
import yaml
import dotenv
import os

dotenv.load_dotenv()

# Get absolute path of the folder containing this file
path_file = os.path.dirname(os.path.abspath(__file__))
# Open the config file 
with open(os.path.join(path_file,"common.yaml"), "r") as f:
    cfg = yaml.safe_load(f)
