import requests
import time
import re
from llm_deploy.interfaces import VastAIInterface

class VastAI(VastAIInterface):
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://console.vast.ai/api/v0/'
        self.headers = {'Accept': 'application/json'}

    def filter_offers(self, offers, filters):
        for filter_func in filters:
            offers = filter(filter_func, offers)
        return list(offers)

    def filter_by_verification(self, offer):
        options = ['verified']
        return offer.get('verification') in options

    def filter_by_internet_speed(self, offer):
        return offer.get('inet_up', 1) > 0 and offer.get('inet_down', 1) > 0

    def filter_by_static_ip(self, offer):
        return offer.get('static_ip', False) == True

    def get_available_offers(
            self, 
            gpu_memory=32000, 
            min_gpu=1, 
            max_gpu=2, 
            disk_space=40, 
            internet_speed=70, 
            result_count=10, 
            public_ip=True
         ):
        url = self.base_url + 'bundles/'
        query_params = {
            "reliability2": {"gte": 0.85},
            "disk_space": {"gte": disk_space},
            "rentable": {"eq": True},
            "num_gpus": {"gte": min_gpu, "lte": max_gpu},
            "gpu_totalram": {"gte": gpu_memory},
            "direct_port_count": {"gte": 1},
            "sort_option": {"0": ["dphtotal", "asc"], "1": ["total_flops", "asc"]},
            "order": [["dphtotal", "asc"], ["total_flops", "asc"]],
            "allocated_storage": disk_space,
            "cuda_max_good": {},
            "extra_ids": [],
            "inet_down":{"gte":internet_speed},
            "type": "ask"
        }
        if public_ip:
            query_params['static_ip'] = {"eq": True}
        response = requests.post(url, headers=self.headers, json=query_params)
        all_offers = response.json()['offers']

        # Set up filters
        filters = [self.filter_by_verification, self.filter_by_internet_speed]
        if public_ip:
            filters.append(self.filter_by_static_ip)

        # Filter offers based on verification status and limit results
        filtered_offers = self.filter_offers(all_offers, filters)
        return filtered_offers[:result_count]

    def create_instance(self, machine_id, disk_space, image="g1ibby/ollama-cloudflared", ports=[]):
        url = self.base_url + f'asks/{machine_id}/?api_key={self.api_key}'
        env_dict = {f"-p {port}:{port}": "1" for port in ports}
        data = {
                "client_id": "me",
                "image": image,
                "env": env_dict,
                "runtype": "args",
                "use_jupyter_lab": False,
                "disk": disk_space,
            }
        response = requests.put(url, headers=self.headers, json=data)
        payload = response.json()
        print(payload)
        if payload.get('success') == False:
            return None
        return payload.get('new_contract')

    def list_instances(self):
        url = self.base_url + f'instances?api_key={self.api_key}'
        response = requests.get(url, headers=self.headers).json()
        return response.get('instances', [])

    def destroy_instance(self, instance_id):
        url = self.base_url + f'instances/{instance_id}/?api_key={self.api_key}'
        response = requests.delete(url, headers=self.headers)
        return response.json()

    def get_instance_logs(self, instance_id, max_attempts=10):
        url = self.base_url + f'instances/request_logs/{instance_id}/?api_key={self.api_key}'
        data = {"tail": "1000"}  # Modify as needed

        for attempt in range(max_attempts):
            response = requests.put(url, headers=self.headers, json=data).json()
            if not response.get('success'):
                print(f"Attempt {attempt + 1}: Failed to get response")
                time.sleep(1)  # Wait for 1 second before retrying
                continue

            logs_url = response.get('result_url')
            if not logs_url:
                print(f"Attempt {attempt + 1}: No logs URL found")
                time.sleep(1)  # Wait for 1 second before retrying
                continue

            log_data = requests.get(logs_url).text
            if "Access Denied" in log_data:
                print(f"Attempt {attempt + 1}: Access Denied. Retrying...")
                time.sleep(1)  # Wait for 1 second before retrying
                continue

            print(f"Logs URL: {logs_url}")
            return log_data.splitlines()

        print("Failed to retrieve logs after multiple attempts.")
        return []

    def retrieve_cloudflared_addr(self, instance_id):
        # Fetch logs
        logs = self.get_instance_logs(instance_id)
        # Join the logs into a single string if they are in a list
        if isinstance(logs, list):
            logs = '\n'.join(logs)

        # Regular expression pattern to find the Cloudflared address
        url_pattern = r'https://[^\s]+\.trycloudflare\.com'
        # Searching for the pattern in the logs
        match = re.search(url_pattern, logs)
        # Return the found URL or None if not found
        return match.group(0) if match else None
