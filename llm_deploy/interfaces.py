from abc import ABC, abstractmethod

class VastAIInterface(ABC):
    """
    Abstract class for interacting with the vast.ai API.
    """

    @abstractmethod
    def get_available_offers(self, gpu_memory=30, min_gpu=1, max_gpu=2, disck_space=40, internet_speed=70):
        """
        Retrieves a list of available offers that match specified criteria.

        :param min_memory: Minimum amount of memory required (in GB).
        :param max_memory: Maximum amount of memory acceptable (in GB).
        :param min_gpu: Minimum number of GPUs required.
        :param max_gpu: Maximum number of GPUs acceptable.
        :param internet_speed: Minimum internet speed required (in Mbps).
        :return: List of offers matching the criteria.
        """
        pass

    @abstractmethod
    def create_instance(self, machine_id, template_id):
        """
        Creates a new instance with given machine and template IDs.

        :param machine_id: The ID of the machine to rent.
        :param template_id: The ID of the template to use for the instance.
        :return: Instance ID of the newly created instance.
        """
        pass

    @abstractmethod
    def list_instances(self):
        """
        Retrieves a list of available instances with their details.

        :return: List of instances including ID, IP address, status, and other metadata.
        """
        pass

    @abstractmethod
    def destroy_instance(self, instance_id):
        """
        Terminates an instance specified by the instance ID.

        :param instance_id: The ID of the instance to be destroyed.
        :return: Confirmation of instance termination.
        """
        pass

    @abstractmethod
    def get_instance_logs(self, instance_id):
        """
        Fetches logs for a specified instance.

        :param instance_id: The ID of the instance from which to retrieve logs.
        :return: Logs of the specified instance.
        """
        pass

class OllamaInstanceInterface(ABC):
    """
    Abstract class for managing an Ollama instance.
    """

    @abstractmethod
    def pull_model(self, model_name):
        """
        Pulls a specific model.

        :param model_name: Name of the model to pull (e.g., "ollama").
        :return: Confirmation of the model pull.
        """
        pass

    @abstractmethod
    def ollama_status(self):
        """
        Returns the status of the Ollama server.

        :return: Status of the server ('running' or 'stopped').
        """
        pass

    @abstractmethod
    def models(self):
        """
        Returns a list of local models.

        :return: List of local models.
        """
        pass

    @abstractmethod
    def test_model(self, model_name):
        """
        Sends a test request to a specific model.

        :param model_name: Name of the model to test.
        :return: True if the test is successful, False otherwise.
        """
        pass
