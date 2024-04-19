import requests

class LiteLLManager:
    def __init__(self, api_url="http://localhost:4000"):
        self.api_url = api_url

    def add_model(self, model_identifier, api_base):
        try:
            response = requests.post(f"{self.api_url}/model/new", json={
                "model_name": model_identifier,
                "litellm_params": {
                    "model": f"ollama/{model_identifier}",
                    "api_base": api_base
                },
                "model_info": {
                    "id": model_identifier,
                }
            })
            if response.status_code != 200:
                print(f"Failed to add model: {response.text}")
        except requests.exceptions.ConnectionError:
            print(f"Failed to connect to {self.api_url}. Skipping model addition to litellm.")

    def get_model_names(self):
        try:
            response = requests.get(f"{self.api_url}/model/info")
            if response.status_code == 200:
                models = response.json().get('data', [])
                return [model['model_name'] for model in models]
            else:
                print(f"Failed to fetch model names. Status code: {response.status_code}")
                return []
        except requests.exceptions.ConnectionError:
            print(f"Failed to connect to {self.api_url}. Skipping model name retrieval from litellm.")
            return []

    def remove_model_by_id(self, model_id):
        try:
            response = requests.post(f"{self.api_url}/model/delete", json={"id": model_id})
            if response.status_code != 200:
                print(f"Failed to remove model: {response.text}")
        except requests.exceptions.ConnectionError:
            print(f"Failed to connect to {self.api_url}. Skipping model removal from litellm.")

    def remove_all_models_by_api_base(self, api_base):
        try:
            response = requests.get(f"{self.api_url}/model/info")
            if response.status_code == 200:
                models = response.json().get('data', [])
                for model in models:
                    litellm_params = model.get('litellm_params', {})
                    if 'api_base' in litellm_params and litellm_params['api_base'] == api_base:
                        try:
                            self.remove_model_by_id(model['model_info']['id'])
                        except requests.exceptions.ConnectionError:
                            print(f"Failed to connect to {self.api_url}. Skipping model removal.")
                    else:
                        print(f"Failed to remove model {model['model_info']['id']}. API base mismatch.")
            else:
                print("Failed to fetch models for removal")
        except requests.exceptions.ConnectionError:
            print(f"Failed to connect to {self.api_url}. Skipping model removal form litellm.")

