import requests
import math
import json

class LLMCalculator:
    def __init__(self):
        self.gguf_quants = {
            "Q3_K_S": 3.5,
            "Q3_K_M": 3.91,
            "Q3_K_L": 4.27,
            "Q4_0": 4.55,
            "Q4_K_S": 4.58,
            "Q4_K_M": 4.85,
            "Q5_0": 5.54,
            "Q5_K_S": 5.54,
            "Q5_K_M": 5.69,
            "Q6_K": 6.59,
            "Q8_0": 8.5,
        }

    def fetch_model_size(self, hf_model: str):
        """
        Tries to fetch the model size from different file types and sources.
        Returns the model size if found, else raises an exception.
        """
        sources = [
            f"https://huggingface.co/{hf_model}/resolve/main/model.safetensors.index.json",
            f"https://huggingface.co/{hf_model}/resolve/main/pytorch_model.bin.index.json"
        ]

        for source in sources:
            try:
                response = requests.get(source)
                response.raise_for_status()  # This will raise an HTTPError for bad responses
                data = response.json()
                model_size = data.get("metadata", {}).get("total_size")
                if model_size and not math.isnan(model_size / 2):
                    return model_size / 2
            except (requests.exceptions.RequestException, ValueError):
                continue  # If there's an error or no size, try the next source

        # If all sources fail, fall back to scraping the model page as the last resort
        return self.scrape_model_page_for_size(hf_model)

    def scrape_model_page_for_size(self, hf_model: str):
        """
        Fallback method to scrape the model's webpage for size information.
        """
        try:
            model_page = requests.get(f"https://huggingface.co/{hf_model}").text
            params_el = model_page.find('data-target="ModelSafetensorsParams"')
            if params_el != -1:
                model_size = json.loads(model_page[params_el:].split('data-props="', 1)[1].split('"', 1)[0])["safetensors"]["total"]
            else:
                params_el = model_page.find('data-target="ModelHeader"')
                model_size = json.loads(model_page[params_el:].split('data-props="', 1)[1].split('"', 1)[0])["model"]["safetensors"]["total"]
            return model_size
        except Exception as e:
            raise ValueError(f"Failed to scrape model size for '{hf_model}'. Error: {e}")

    def model_config(self, hf_model: str) -> dict:
        """
        Retrieves the model configuration JSON from the Hugging Face API and determines the model size.
        """
        config_response = requests.get(f"https://huggingface.co/{hf_model}/raw/main/config.json")
        config = config_response.json()

        # Use the new fetch_model_size method to get model size
        model_size = self.fetch_model_size(hf_model)
        config["parameters"] = model_size
        return config

    def input_buffer(self, context: int, model_config: dict, bsz: int) -> float:
        """
        Calculates the size of the input buffer based on the context size, model configuration, and batch size.
        """
        # Calculation taken from github:ggerganov/llama.cpp/llama.cpp:11248
        inp_tokens = bsz
        inp_embd = model_config["hidden_size"] * bsz
        inp_pos = bsz
        inp_kq_mask = context * bsz
        inp_k_shift = context
        inp_sum = bsz

        return inp_tokens + inp_embd + inp_pos + inp_kq_mask + inp_k_shift + inp_sum

    def compute_buffer(self, context: int, model_config: dict, bsz: int) -> float:
        """
        Estimates the size of the compute buffer based on the context size, model configuration, and batch size.
        """
        if bsz != 512:
            raise Exception("Batch size other than 512 is currently not supported for the compute buffer")

        return (context / 1024 * 2 + 0.75) * model_config["num_attention_heads"] * 1024 * 1024

    def kv_cache(self, context: int, model_config: dict, fp8_cache: bool) -> float:
        """
        Calculates the size of the key-value cache based on the context size, model configuration, and FP8 caching flag.
        """
        n_gqa = model_config["num_attention_heads"] / model_config["num_key_value_heads"]
        n_embd_gqa = model_config["hidden_size"] / n_gqa
        n_elements = n_embd_gqa * (model_config["num_hidden_layers"] * context)
        size = 2 * n_elements

        if fp8_cache:
            return size
        else:
            return size * 2

    def context_size(self, context: int, model_config: dict, bsz: int, fp8_cache: bool) -> float:
        """
        Combines the input buffer size, key-value cache size, and compute buffer size to estimate the total memory required for the context.
        """
        input_buffer_size = self.input_buffer(context, model_config, bsz)
        kv_cache_size = self.kv_cache(context, model_config, fp8_cache)
        compute_buffer_size = self.compute_buffer(context, model_config, bsz)

        return float(f"{input_buffer_size + kv_cache_size + compute_buffer_size:.2f}")

    def model_size(self, model_config: dict, quant_size: str) -> float:
        """
        Calculates the size of the model based on the number of parameters and the quantization size.
        """
        bpw = self.gguf_quants[quant_size]
        return float(f"{model_config['parameters'] * bpw / 8:.2f}")

    def extract_model_info(self, model_input: str) -> tuple:
        model_parts = model_input.split(":")
        model_name_parts = model_parts[0].split("-")

        # Identify the quant_size directly from model_input without needing to split again
        quant_size = model_parts[1].split("-")[-1].upper()

        # Reconstruct the model name with the size detail (e.g., 8x7b) included but without the quant size or additional versioning info
        size_detail = model_parts[1].split("-")[0]  # Assuming the size detail is always the first part after colon
        model_name = f"{model_name_parts[0]}-{size_detail}"

        # Make a request to the Hugging Face API
        url = f"https://huggingface.co/api/quicksearch?type=model&q={model_name}"
        response = requests.get(url)
        data = response.json()

        # Extract the full model name from the first search result
        models = data["models"]
        if len(models) > 0:
            full_model_name = models[0]["id"]
        else:
            raise ValueError(f"No models found for '{model_name}' in the Hugging Face API.")

        return full_model_name, quant_size

    def calculate_sizes(self, model: str, quant_size: str, context: int, bsz: int = 512, fp8_cache: bool = False) -> tuple:
        """
        Orchestrates the overall calculation by calling the necessary methods based on the selected quantization size.
        Returns the model size, context size, and total size in gigabytes (GB).
        """
        model_config = self.model_config(model)

        if quant_size not in self.gguf_quants:
            raise Exception(f"Unsupported quantization size: {quant_size}")

        model_size = self.model_size(model_config, quant_size)
        context_size = self.context_size(context, model_config, bsz, fp8_cache)
        total_size = (model_size + context_size) / 1e9

        return model_size / 1e9, context_size / 1e9, total_size

    def calculate(self, model_input: str, context: int) -> tuple:
        """
        Calculates the model size, context size, and total size based on the model name input and context size.
        """
        full_model_name, quant_size = self.extract_model_info(model_input)
        model_size, context_size, total_size = self.calculate_sizes(full_model_name, quant_size, context)
        return model_size, context_size, total_size
