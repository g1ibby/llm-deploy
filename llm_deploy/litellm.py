import requests

class LiteLLManager:
    def __init__(self, api_url="http://localhost:4000"):
        self.api_url = api_url

    # Create new model in LiteLLM proxy server
    # model_identifier: Identifier of the model withouth the "ollama/" prefix
    # api_base: Base URL of the ollama API server
    def add_model(self, model_identifier, api_base):
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
            raise Exception(f"Failed to add model: {response.text}")

    # Get list of models in LiteLLM proxy server
    # Returns a list of model names with the "ollama/" prefix
    def get_model_names(self):
        response = requests.get(f"{self.api_url}/model/info")
        models = response.json().get('data', [])
        return [model['model_name'] for model in models]

    def remove_model_by_id(self, model_id):
        response = requests.post(f"{self.api_url}/model/delete", json={"id": model_id})
        if response.status_code != 200:
            raise Exception(f"Failed to remove model: {response.text}")

    def remove_all_models_by_api_base(self, api_base):
        response = requests.get(f"{self.api_url}/model/info")
        if response.status_code == 200:
            models = response.json().get('data', [])
            for model in models:
                litellm_params = model.get('litellm_params', {})
                if 'api_base' in litellm_params and litellm_params['api_base'] == api_base:
                    try:
                        self.remove_model_by_id(model['model_info']['id'])
                    except Exception as e:
                        print(f"Failed to remove model {model['model_info']['id']}: {e}")
                        continue
        else:
            raise Exception("Failed to fetch models for removal")
