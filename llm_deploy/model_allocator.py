from llm_deploy.ollama import retrieve_model_size

PRIORITY_MAP = {
    'high': 2,
    'low': 0
}

class ModelAllocator:
    """
    This class allocates Large Language Models (LLMs) to available machines with GPUs.
    It optimizes the allocation based on the model's priority, size, and machine characteristics
    like GPU RAM, total flops, price per hour, and number of GPUs.

    High-priority models are allocated to machines where their combined size does not exceed
    the machine's GPU RAM. Low-priority models are allocated to any machine with enough space
    for the individual model.

    The allocator uses the `vast` object to retrieve machine offers and caches the GPU RAM of
    each machine for efficient space management.

    Methods:
    - `load_desired_models`: Loads and sorts the desired models based on priority and size.
    - `get_available_offers`: Retrieves a list of available machines based on required GPU memory.
    - `allocate_models`: Allocates models to machines based on priority and available resources.
    - `find_suitable_machine`: Finds the most suitable machine for a given model.
    - `allocate_model_to_machine`: Allocates a model to a specific machine.
    - `can_allocate`: Checks if a model can be allocated to a machine with available space.
    - `calculate_required_gpu_memory`: Calculates the required GPU memory for a model.
    - `update_available_space`: Updates the available GPU RAM of a machine after allocation.
    """

    # Constant for additional RAM required per model (in GB)
    MODEL_RAM_OVERHEAD = 1

    def __init__(self, vast, llms_config):
        self.allocations = {}  # Maps machine ID to list of allocated models
        self.available_space = {}  # Tracks available GPU RAM for each machine
        self.gpu_ram_cache = {}  # Caches the GPU RAM of each machine
        self.machines = {} # Maps machine ID to machine object
        self.vast = vast  # Vast object to interact with machine offers
        self.llms_config = llms_config  # Configuration object for LLMs
        self.desired_models = self.load_desired_models()  # List of desired models

    def load_desired_models(self):
        """
        Retrieves a list of LLMs from configuration, adds their size information,
        and sorts them by priority (high first) and size (larger first).

        Returns:
            List[dict]: Sorted list of models with their properties.
        """
        models = self.llms_config.get_models()
        for model in models:
            model_name = model['model']
            print(f"Retrieving size for model: {model_name}")
            model_size = retrieve_model_size(model_name)
            model_size_mb = model_size * 1024  # Convert GB to MB

            model['size'] = model_size_mb

        # Sorts models by priority (high first) and size (larger first)
        return sorted(models, key=lambda x: (-PRIORITY_MAP[x['priority']], -x['size']))

    def get_available_offers(self, gpu_memory, min_gpu=1, max_gpu=2, disk_space=40, internet_speed=100, result_count=10, public_ip=True):
        # Your existing implementation
        # Make sure to update self.gpu_ram_cache with the gpu_totalram of each machine
        machines = self.vast.get_available_offers(gpu_memory, min_gpu, max_gpu, disk_space, internet_speed, result_count, public_ip)
        for machine in machines:
            self.gpu_ram_cache[machine['id']] = machine['gpu_totalram']
        return machines

    def allocate_models(self):
        for model in self.desired_models:
            machine = self.find_suitable_machine(model)
            if not machine:
                print(f"Failed to allocate model: {model['model']}")
                continue
            machine_id = machine['id']
            self.machines[machine_id] = machine
            self.allocate_model_to_machine(model, machine_id)

        return self.allocations, self.machines

    def find_suitable_machine(self, model):
        for machine_id, space in self.available_space.items():
            if self.can_allocate(model, space, self.allocations.get(machine_id, [])):
                return self.machines[machine_id]

        required_gpu_memory = self.calculate_required_gpu_memory(model)
        machines = self.get_available_offers(gpu_memory=required_gpu_memory)

        # Filtering machines with more than two GPUs
        suitable_machines = [m for m in machines if m['num_gpus'] <= 2]

        # Selecting the machine based on total_flops and price
        suitable_machines.sort(key=lambda m: (-m['total_flops'], m['dph_total']))

        return suitable_machines[0] if suitable_machines else None

    def allocate_model_to_machine(self, model, machine_id):
        if machine_id in self.allocations:
            self.allocations[machine_id].append(model)
        else:
            self.allocations[machine_id] = [model]
        self.update_available_space(machine_id, model)

    def can_allocate(self, model, available_ram, models_on_machine):
        total_size_on_machine = sum(m['size'] + ModelAllocator.MODEL_RAM_OVERHEAD for m in models_on_machine)
        new_total_size = total_size_on_machine + model['size'] + ModelAllocator.MODEL_RAM_OVERHEAD

        if model['priority'] == 'high':
            return new_total_size <= available_ram
        else:
            return (model['size'] + ModelAllocator.MODEL_RAM_OVERHEAD) <= available_ram

    def calculate_required_gpu_memory(self, model):
        if model['priority'] == 'high':
            high_priority_sizes = [m['size'] for m in self.desired_models if m['priority'] == 'high']
            return sum(high_priority_sizes)
        else:
            return model['size']

    def update_available_space(self, machine_id, model):
        if machine_id in self.available_space:
            self.available_space[machine_id] -= (model['size'] + ModelAllocator.MODEL_RAM_OVERHEAD)
        else:
            # Use the cached gpu_totalram for the new machine
            self.available_space[machine_id] = self.gpu_ram_cache[machine_id] - (model['size'] + ModelAllocator.MODEL_RAM_OVERHEAD)

