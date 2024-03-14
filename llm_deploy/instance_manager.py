import time
import requests

from llm_deploy.ollama import OllamaInstance

class InstanceManager:
    def __init__(self, vast, storage, litellm):
        self.vast = vast
        self.storage = storage
        self.litellm = litellm

    def create(self, offer_id, disk_space, public_ip=True):
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

    def destroy_all(self):
        """
        Destroy all the instances based on state.json file.
        """
        instances = self.instances()
        for instance in instances:
            self.destroy_instance(instance['id'])

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

        r = self.vast.destroy_instance(instance_id)
        if r['success']:
            instances = self.vast.list_instances()
            self.storage.sync_instances([inst['id'] for inst in instances])
            self.litellm.remove_all_models_by_api_base(instance['ollama_addr'])
            print("Instance destroyed successfully.")

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

    def get_instance_logs(self, instance_id, max_logs=30):
        """
        Retrieve logs for a specific instance.
        :param instance_id: Instance ID
        :param max_logs: Maximum number of logs to retrieve
        :return: List of logs
        """
        logs = self.vast.get_instance_logs(instance_id)
        return logs[-max_logs:]

