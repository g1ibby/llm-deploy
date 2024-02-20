import os

def load_config():
    config_path = os.path.expanduser('~/.vast_api_key')
    api_key = None  # Initialize variable to hold the API key

    # Check if the file exists and read the key from it
    if os.path.isfile(config_path):
        with open(config_path, 'r') as file:
            api_key = file.read().strip()

    # Ensure VAST_API_KEY is obtained either from file or environment variable
    vast_api_key = api_key if api_key else os.environ.get('VAST_API_KEY', '')

    # Always read LITELLM_API_URL from the environment variable
    litellm_api_url = os.environ.get('LITELLM_API_URL', 'http://localhost:4000')

    return {
        'VAST_API_KEY': vast_api_key,
        'LITELLM_API_URL': litellm_api_url
    }
