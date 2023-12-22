import os

def load_config():
    config_path = os.path.expanduser('~/.vast_api_key')

    # Check if the file exists and read the key from it
    if os.path.isfile(config_path):
        with open(config_path, 'r') as file:
            key = file.read().strip()
            if key:
                return {'VAST_API_KEY': key}

    # If the file doesn't exist or is empty, read the key from the environment variable
    return {'VAST_API_KEY': os.environ.get('VAST_API_KEY')}
