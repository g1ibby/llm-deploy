import json
import pytest
from unittest.mock import patch
from llm_deploy.ollama import ModelAllocator, retrieve_model_size

# Utility function to load mock data
def load_mock_data(file_name):
    with open(f"tests/mocks/{file_name}", "r") as file:
        return json.load(file)

# Test for allocate_models
def test_allocate_models():
    vast_mock = Mock()
    llms_config_mock = Mock(get_models=Mock(return_value=[
        {'model': 'ModelA', 'priority': 'high', 'size': 8 * 1024},  # 8GB in MB
        {'model': 'ModelB', 'priority': 'low', 'size': 12 * 1024}   # 12GB in MB
    ]))

    allocator = ModelAllocator(vast=vast_mock, llms_config=llms_config_mock)

    # Mock data files corresponding to different GPU memory sizes
    mock_files = {
        8 * 1024: 'case_8GB.json',
        12 * 1024: 'case_12GB.json',
        # Add more mappings as needed
    }

    # Patch the get_available_offers method
    with patch.object(vast_mock, 'get_available_offers', side_effect=lambda gpu_memory, **kwargs: load_mock_data(mock_files[gpu_memory])):
        allocations, machines = allocator.allocate_models()

        # Assertions go here
        # Example: assert len(allocations) == expected_number_of_allocations
        # More detailed assertions can be added based on the expected outcome

