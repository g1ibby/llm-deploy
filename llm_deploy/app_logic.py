from llm_deploy.vastai import VastAI
import time
from llm_deploy.ollama import OllamaInstance
from llm_deploy.utils import print_pull_status
from llm_deploy.storage_manager import StorageManager
from llm_deploy.litellm import LiteLLManager
from llm_deploy.llms_config import LLMsConfig
from llm_deploy.model_allocator import ModelAllocator

class AppLogic:
    def __init__(self, vast_api_key):
        """
        Initialize the AppLogic class with the VastAI API key.
        """
        self.vast_api_key = vast_api_key
        self.vast = VastAI(vast_api_key)
        self.storage = StorageManager()
        self.llms_config = LLMsConfig()

    def apply_llms_config(self):
        """
        Apply the LLMs configuration.
        """
        model_allocator = ModelAllocator(self.vast, self.llms_config)
        allocated_models, machines = model_allocator.allocate_models()
        print("Allocated Models:")
        print(allocated_models)
        print("Machines:")
        print(machines)

        # creating instances
        for machine_id, models in allocated_models.items():
            instance_id, _ = self.create_instance(machine_id, 70, True)
            if not instance_id:
                print("Failed to create instance.")
                return None
            # will pull models for this instance 
            for model in models:
                if not self.pull_model(model['model'], instance_id):
                    print("Failed to pull model.")
                    return None

        return None

    def destroy(self):
        """
        Destroy all the instances based on state.json file.
        """
        instances = self.instances()
        for instance in instances:
            self.destroy_instance(instance['id'])

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

    def get_instance_address(self, instance):
        public_ip = instance.get('public_ipaddr', 'N/A')
        if public_ip == 'N/A':
            return None
        # clean public ip new line and spaces
        public_ip = public_ip.replace('\n', '').strip()
        ports = instance.get('ports', {})
        if not ports:
            return None

        port_info = next(iter(ports.values()), [{}])[0]
        host_port = port_info.get('HostPort', '')
        return f"http://{public_ip}:{host_port}" if host_port else public_ip

    def monitor_instance_status(self, instance_id, retry_count=30, delay=10):
        for attempt in range(retry_count): 
            instances = self.instances()
            chosen_instance = next((inst for inst in instances if inst['id'] == instance_id), None)

            if chosen_instance is None:
                print("Instance not found.")
                time.sleep(delay)
                continue

            actual_status = (chosen_instance.get('actual_status') or '').lower()
            intended_status = (chosen_instance.get('intended_status') or '').lower()
            cur_state = (chosen_instance.get('cur_state') or '').lower()
            status_msg = (chosen_instance.get('status_msg') or '').lower()

            print(f"Instance Status:\nActual: {actual_status}\nIntended: {intended_status}\nCurrent: {cur_state}\nMessage: {status_msg}")

            if actual_status == "running" and intended_status == "running" and cur_state == "running":
                address = self.get_instance_address(chosen_instance)
                if address:
                    return chosen_instance, address
                else:
                    print("Failed to retrieve instance address")
            elif "error" in status_msg:
                print("Error encountered with the instance.")
                return None, None

            if attempt < retry_count - 1:
                print(f"Retrying to get instance status... (Attempt {attempt + 1}/{retry_count})")
                time.sleep(delay)

        print(f"Instance did not reach the 'running' status after {retry_count} attempts.")
        return None, None

    def cloudflared(self, instance_id):
        cloudflared_addr = None
        for attempt in range(10):
            cloudflared_addr = self.vast.retrieve_cloudflared_addr(instance_id)
            if cloudflared_addr:
                break
            else:
                print("Waiting for Cloudflared address to be available...")
                time.sleep(5)
        if not cloudflared_addr:
            print("Failed to retrieve Cloudflared address.")
            print("Destroying instance...")
            self.vast.destroy_instance(instance_id)
            return None
        return cloudflared_addr

    def create_instance(self, offer_id, disk_space, public_ip=True):
        # Create an instance
        image = "g1ibby/ollama-cloudflared:latest"
        ports = []
        if public_ip:
            image = "ollama/ollama:latest"
            ports = [11434]
        instance_id = self.vast.create_instance(offer_id, image=image, ports=ports, disk_space=disk_space)
        print(f"Created Instance with ID: {instance_id}")

        # Monitor instance status
        chosen_instance, _ = self.monitor_instance_status(instance_id)
        if not chosen_instance:
            print("Instance with ollama creation failed.")
            print("Destroying instance...")
            self.vast.destroy_instance(instance_id)
            return None

        print(f"Instance Status: {chosen_instance['actual_status']}")

        # Get Ollama address
        ollama_addr = self.cloudflared(instance_id) if not public_ip else self.get_instance_address(chosen_instance)
        if not ollama_addr:
            print("Failed to retrieve Ollama address.")
            self.vast.destroy_instance(instance_id)
            return None

        # Save instance details
        self.storage.save_instance(instance_id, {"ollama_addr": ollama_addr})
        print(f"Ollama address: {ollama_addr}")

        # Create an instance of the OllamaInstance class
        ollama_instance = OllamaInstance(ollama_addr)
        # Wait for Ollama server to be 'running'
        ollama_running = False
        for attempt in range(10):
            ollama_status = ollama_instance.ollama_status()
            print(f"Checking Ollama Server Status: {ollama_status}")
            if ollama_status == "running":
                ollama_running = True
                break
            else:
                print("Waiting for Ollama server to start...")
                time.sleep(10)
                print(f"Retrying to get Ollama server status... (Attempt {attempt + 1}/10)")

        if not ollama_running:
            print("Ollama server did not reach the 'running' status after 10 attempts.")
            print("Destroying instance...")
            self.vast.destroy_instance(instance_id)
            return None

        print("Ollama Server Status: Running")
        return instance_id, ollama_addr

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
        instance_id, ollama_addr = self.create_instance(offer_id, disk_space, public_ip)
        if not instance_id:
            return False

        # Pull the model using the existing pull_model function
        if not self.pull_model(model, instance_id):
            print("Failed to pull model.")
            self.vast.destroy_instance(instance_id)
            return False

        # Test a model and print responses
        ollama_instance = OllamaInstance(ollama_addr)
        print(f"Testing model: {model}")
        test_result = ollama_instance.test_model(model)
        print(f"Test Result: {test_result}")
        return test_result

    def pull_model(self, model_name: str, instance_id: int):
        """
        Pull a model from the Ollama server.
        :param model_name: Model name
        :param instance_id: Instance ID
        :return: Pull status
        """
        print(f"Pulling model: {model_name}")
        # Get the instance address
        instance = self.get_instance_by_id(instance_id)
        if not instance:
            print("Instance not found.")
            return False
        ollama_addr = instance.get('ollama_addr')
        if not ollama_addr:
            print("Ollama address not found.")
            return False

        # Create an instance of the OllamaInstance class
        ollama_instance = OllamaInstance(ollama_addr)
        # Pull a model and print updates
        model_pull_generator = ollama_instance.pull_model(model_name)
        print_pull_status(model_pull_generator)
        litellm_instance = LiteLLManager()
        litellm_instance.add_model(model_name, ollama_addr)

        return True

    def remove_model(self, model_name: str, instance_id: int):
        """
        Remove a model from the Ollama server.
        :param model_name: Model name
        :param instance_id: Instance ID
        :return: Removal status
        """
        # Get the instance address
        instance = self.get_instance_by_id(instance_id)
        if not instance:
            print("Instance not found.")
            return False
        ollama_addr = instance.get('ollama_addr')
        if not ollama_addr:
            print("Ollama address not found.")
            return False

        # Create an instance of the OllamaInstance class
        ollama_instance = OllamaInstance(ollama_addr)
        # Remove a model and print updates
        return ollama_instance.remove_model(model_name)

    def models(self):
        """
        List all models.
        :return: List of models
        """
        instances = self.instances()
        models = []
        for instance in instances:
            if instance['ollama_addr'] != '':
                ollama_instance = OllamaInstance(instance['ollama_addr'])
                list_of_models = ollama_instance.models()
                for model in list_of_models:
                    model['instance_id'] = instance['id']
                models += list_of_models
        return models

    def instances(self):
        """
        List all instances.
        :return: List of instances
        """
        instances = self.vast.list_instances()
        self.storage.sync_instances([inst['id'] for inst in instances])
        # Inject Cloudflared address into the instances
        for instance in instances:
            storage_instance = self.storage.get_instance(instance['id'])
            instance['ollama_addr'] = storage_instance.get('ollama_addr', '')
        return instances

    def get_instance_by_id(self, instance_id: int):
        """
        Retrieve a specific instance by its ID.
        :param instance_id: Instance ID
        :return: Instance details or None if not found
        """
        instances = self.instances()
        chosen_instance = next((inst for inst in instances if inst['id'] == instance_id), None)
        # Inject list of models in the chosen_instance based on ollama_addr
        if chosen_instance and chosen_instance['ollama_addr'] != '':
            ollama_instance = OllamaInstance(chosen_instance['ollama_addr'])
            chosen_instance['models'] = ollama_instance.models()
        return chosen_instance

    def destroy_instance(self, instance_id):
        """
        Destroy a specific instance by its ID.
        :param instance_id: Instance ID
        :return: Result of the destruction operation
        """
        instance = self.get_instance_by_id(instance_id)

        litellm_instance = LiteLLManager()
        litellm_instance.remove_all_models_by_api_base(instance['ollama_addr'])
        r = self.vast.destroy_instance(instance_id)
        instances = self.vast.list_instances()
        self.storage.sync_instances([inst['id'] for inst in instances])
        print(r)

    def get_instance_logs(self, instance_id, max_logs=30):
        """
        Retrieve logs for a specific instance.
        :param instance_id: Instance ID
        :param max_logs: Maximum number of logs to retrieve
        :return: List of logs
        """
        logs = self.vast.get_instance_logs(instance_id)
        return logs[-max_logs:]

