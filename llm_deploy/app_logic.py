from llm_deploy.vastai import VastAI
from llm_deploy.ollama import OllamaInstance
from llm_deploy.storage_manager import StorageManager
from llm_deploy.litellm import LiteLLManager
from llm_deploy.llms_config import LLMsConfig
from llm_deploy.model_allocator import ModelAllocator
from llm_deploy.instance_manager import InstanceManager
from llm_deploy.model_manager import ModelManager

class AppLogic:
    def __init__(self, vast_api_key, litellm_api_url):
        """
        Initialize the AppLogic class with the VastAI API key.
        """
        self.vast_api_key = vast_api_key
        self.vast = VastAI(vast_api_key)
        self.storage = StorageManager()
        self.llms_config = LLMsConfig()
        self.litellm = LiteLLManager(litellm_api_url)
        self.instance = InstanceManager(self.vast, self.storage, self.litellm)
        self.model = ModelManager(self.litellm, self.storage)

    def apply_llms_config(self):
        """
        Apply the LLMs configuration.
        """
        model_allocator = ModelAllocator(self.vast, self.llms_config)
        allocated_models, machines = model_allocator.allocate_models()
        self.log_machine_details(allocated_models, machines)

        models_size = self._calculate_models_size(allocated_models)
        # creating instances
        for machine_id, models in allocated_models.items():
            machine_disk_space = (models_size[machine_id] + 5000) / 1024
            instance_id, _ = self.instance.create(machine_id, machine_disk_space, True)
            if not instance_id:
                print("Failed to create instance.")
                return None
            # will pull models for this instance 
            for model in models:
                if not self.model.pull(model['model'], instance_id):
                    print("Failed to pull model.")
                    return None

        return None

    def log_machine_details(self, allocated_models, machines):
        print("Machine Details and Allocated Models\n")
        for machine_id, models in allocated_models.items():
            machine = machines[machine_id]
            print(f"Machine ID: {machine_id}")
            print(f"Price (per hour): ${machine['dph_total']}")
            print(f"GPU: {machine['gpu_name']} | Count: {machine['num_gpus']} | Memory: {machine['gpu_ram']} MB per GPU, Total: {machine['gpu_total_ram']} MB")
            print(f"Internet Speed: Up {machine['inet_up']} Mbps / Down {machine['inet_down']} Mbps")
            print("Allocated Models:")
            for model in models:
                print(f"  - Name: {model['name']}, Model: {model['model']}, Size: {model['size']} MB")
            print("\n" + "-"*50 + "\n")

    def _calculate_models_size(self, models_dict):
        # Dictionary to hold the total size of models for each key
        total_sizes = {}
        
        # Iterate through each key and its list of models in the dictionary
        for key, models in models_dict.items():
            # Calculate the sum of sizes for the current list of models
            total_size = sum(model['size'] for model in models)
            # Assign the total size to the corresponding key in the result dictionary
            total_sizes[key] = total_size
        
        return total_sizes

    def get_offers(self, gpu_memory, disk_space, public_ip=True):
        """
        Retrieve offers based on the specified GPU memory.
        :param gpu_memory: GPU memory in GB
        :param disk_space: Disk space in GB
        :param public_ip: Whether to use public IP or not
        :return: List of offers
        """
        gpu_memory_mb = gpu_memory * 1024  # Convert GB to MB
        offers = self.vast.get_available_offers(gpu_memory=gpu_memory_mb, disk_space=disk_space, public_ip=public_ip)
        return offers

    def run_model(self, model, offer_id, disk_space, public_ip=True):
        """
        Run a specified model with given GPU memory.
        :param model: Model name
        :offer_id: Offer ID
        :param disk_space: Disk space in GB
        :public_ip: Whether to use public IP or not
        :return: Result of the model run
        """
        # Create an instance and prepare it
        instance_id, ollama_addr = self.instance.create(offer_id, disk_space, public_ip)
        if not instance_id:
            return False

        # Pull the model using the existing pull_model function
        if not self.model.pull(model, instance_id):
            print("Failed to pull model.")
            self.vast.destroy_instance(instance_id)
            return False

        # Test a model and print responses
        ollama_instance = OllamaInstance(ollama_addr)
        print(f"Testing model: {model}")
        test_result = ollama_instance.test_model(model)
        print(f"Test Result: {test_result}")
        return test_result

