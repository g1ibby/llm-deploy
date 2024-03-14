from llm_deploy.ollama import OllamaInstance
from llm_deploy.utils import print_pull_status

class ModelManager:
    def __init__(self, litellm, storage):
        self.litellm = litellm
        self.storage = storage

    def pull(self, model_name: str, instance_id: int):
        """
        Pull a model from the Ollama server.
        :param model_name: Model name
        :param instance_id: Instance ID
        :return: Pull status
        """
        print(f"Pulling model: {model_name}")
        # Get the instance address
        instance = self.storage.get_instance(instance_id)
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
        self.litellm.add_model(model_name, ollama_addr)

        return True

    def remove_model(self, model_name: str, instance_id: int):
        """
        Remove a model from the Ollama server.
        :param model_name: Model name
        :param instance_id: Instance ID
        :return: Removal status
        """
        # Get the instance address
        instance = self.storage.get_instance(instance_id)
        if not instance:
            print("Instance not found.")
            return False
        ollama_addr = instance.get('ollama_addr')
        if not ollama_addr:
            print("Ollama address not found.")
            return False

        # Create an instance of the OllamaInstance class
        ollama_instance = OllamaInstance(ollama_addr)
        self.litellm.remove_model_by_id(model_name)
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

