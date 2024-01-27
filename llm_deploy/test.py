import json
from llm_deploy.vastai import VastAI
from llm_deploy.config import load_config

def record_api_case(vast, gpu_memory_gb, filename):
    response = vast.get_available_offers(
        gpu_memory=gpu_memory_gb * 1024,  # Convert GB to MB
        min_gpu=1, 
        max_gpu=2
    )
    with open(f"tests/mocks/{filename}.json", "w") as file:
        json.dump(response, file, indent=4)

config = load_config()

# Initialize the business logic with necessary services
vast = VastAI(config['VAST_API_KEY'])  # Initialize your Vast object

# List of GPU memory sizes in GB
gpu_memory_sizes = [8, 12, 16, 24, 32, 48, 56]

# Record each case
for gpu_memory_gb in gpu_memory_sizes:
    filename = f"case_{gpu_memory_gb}GB"
    record_api_case(vast, gpu_memory_gb, filename)

