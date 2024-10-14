import os
import yaml
import time
from ultralytics import YOLO
from .modules.layoutlmv3.model_init import Layoutlmv3_Predictor

def load_models_and_params():
    """
    Loads model configurations from a YAML file, ensures that necessary models exist, downloads them if missing,
    and initializes the models for further usage.

    This function performs the following steps:
    1. Reads the model configuration file (`model_configs.yaml`) located in the `configs` folder.
    2. Constructs absolute paths for model weights based on the current directory.
    3. Verifies the existence of model files. If the models are missing, it downloads the required models from Hugging Face.
    4. Initializes and returns the MFD and LayoutLMv3 models along with their configurations (image size, confidence threshold, IOU threshold).

    Returns:
        tuple: 
            - mfd_model (YOLO): The YOLO model initialized with the MFD weights.
            - layout_model (Layoutlmv3_Predictor): The LayoutLMv3 model initialized with the Layout weights.
            - img_size (int): The image size to be used for the model.
            - mfd_conf_thres (float): Confidence threshold for MFD model detection.
            - mfd_iou_thres (float): IOU threshold for MFD model detection.
    """
    
    # Start the timer
    start_time = time.time()
    
    # Get the current script's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the model configuration file
    config_path = os.path.join(current_dir, 'configs', 'model_configs.yaml')

    # Load model configurations from the YAML file
    with open(config_path, 'r') as file:
        model_configs = yaml.safe_load(file)

    # Update the paths for model weights to be absolute
    model_configs['model_args']['mfd_weight'] = os.path.join(current_dir, model_configs['model_args']['mfd_weight'])
    model_configs['model_args']['layout_weight'] = os.path.join(current_dir, model_configs['model_args']['layout_weight'])

    # Path to the LayoutLMv3 model configuration
    layout_model_config = os.path.join(current_dir, "modules", "layoutlmv3", "layoutlmv3_base_inference.yaml")

    # Check if model weights exist, if not, download them from Hugging Face
    if not os.path.exists(model_configs['model_args']['mfd_weight']) or not os.path.exists(model_configs['model_args']['layout_weight']):
        from huggingface_hub import snapshot_download

        # Download the Layout model weights
        snapshot_download(
            repo_id="opendatalab/PDF-Extract-Kit",
            allow_patterns="models/Layout/*",
            local_dir=os.path.join(current_dir, 'models', 'Layout')
        )

        # Download the MFD model weights
        snapshot_download(
            repo_id="opendatalab/PDF-Extract-Kit",
            allow_patterns="models/MFD/*",
            local_dir=os.path.join(current_dir, 'models', 'MFD')
        )

    # Load the models with the provided weights
    mfd_model = YOLO(model_configs['model_args']['mfd_weight'])
    layout_model = Layoutlmv3_Predictor(model_configs['model_args']['layout_weight'], layout_model_config)

    # Extract additional model parameters from the configuration file
    img_size = model_configs['model_args']['img_size']
    mfd_conf_thres = model_configs['model_args']['conf_thres']
    mfd_iou_thres = model_configs['model_args']['iou_thres']
    
    # End the timer and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Model loading and initialization took {elapsed_time:.2f} seconds.")
    
    # Return the initialized models and additional parameters
    return mfd_model, layout_model, img_size, mfd_conf_thres, mfd_iou_thres
