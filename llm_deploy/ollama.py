import requests
import json
from bs4 import BeautifulSoup
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

def retrieve_model_size(full_model_name):
    # Check for the presence of '/' to determine the format
    if '/' in full_model_name:
        base_path = full_model_name.split(':')[0]  # For models donwloaded on a personal account
    else:
        base_path = "library/" + full_model_name.split(':')[0]  # For models downloaded on the library

    url = f"https://ollama.ai/{base_path}/tags"
    print(url)
    html_content = _download_html(url)
    model_list = _extract_models(html_content)
    print(model_list)
    return _get_model_size(model_list, full_model_name)

def _download_html(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # This will raise an exception for HTTP errors
        return response.text
    except requests.exceptions.RequestException as e:
        return str(e)

def _extract_models(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    models = []
    
    for div in soup.find_all("div", class_="flex-1"):
        name = div.find("div", class_="break-all font-medium text-gray-900 group-hover:underline").text.strip()
        size_span = div.find("div", class_="flex items-baseline space-x-1 text-[13px] text-neutral-500").span
        if size_span:
            size = size_span.text.split('•')[1].strip()
        else:
            size = "N/A"
        models.append({'name': name, 'size': size})
    
    return models

def _get_model_size(model_list, full_model_name):
    model_name = full_model_name.split(':')[1]

    for model in model_list:
        if model['name'] == model_name:
            size_gb = float(model['size'].replace('GB', ''))
            return size_gb

    return None
