import requests
import json
from llm_deploy.interfaces import OllamaInstanceInterface

class OllamaInstance(OllamaInstanceInterface):
    def __init__(self, address):
        self.address = address

    def pull_model(self, model_name):
        data = {"name": model_name}
        response = requests.post(f"{self.address}/api/pull", json=data, stream=True)
        return self._process_stream(response)

    def ollama_status(self):
        try:
            response = requests.get(self.address)
        except Exception as e:
            print(f"Error of getting ollama status: {e}")
            return None
        if response.text == "Ollama is running":
            return "running"
        else:
            return "stopped"

    def models(self):
        response = requests.get(f"{self.address}/api/tags")
        return response.json()["models"]

    def test_model(self, model_name):
        data = {"model": model_name, "prompt": "Who is the president of the United States?"}
        response = requests.post(f"{self.address}/api/generate", json=data, stream=True)
        return self._process_test_stream(response)

    def remove_model(self, model_name):
        data = {"name": model_name}
        response = requests.delete(f"{self.address}/api/delete", json=data)
        # check if the response is 200 return True
        if response.status_code == 200:
            return True
        return False

    def _process_stream(self, response):
        if response.status_code != 200:
            yield {"error": response.text}

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                status = json.loads(decoded_line)
                yield status

    def _process_test_stream(self, response):
        if response.status_code != 200:
            return False

        last_response_done = False
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                status = json.loads(decoded_line)
                if "done" in status:
                    last_response_done = status["done"]
        return last_response_done
