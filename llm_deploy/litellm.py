import yaml
import os

class LiteLLManager:
    def __init__(self, config_file='config.yaml'):
        self.config_file = config_file
        self.config_data = {'model_list': []}
        self._load_config()

    def _load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as file:
                self.config_data = yaml.safe_load(file) or {'model_list': []}

    def _save_config(self):
        with open(self.config_file, 'w') as file:
            yaml.dump(self.config_data, file, default_flow_style=False)

    def add_model(self, model_identifier, api_base):
        base_model_name = model_identifier
        full_model_name = base_model_name
        count = 1

        for model in self.config_data['model_list']:
            if model['model_name'].startswith(base_model_name):
                if model['litellm_params']['api_base'] == api_base:
                    # Model with same identifier and api_base exists, do nothing
                    return
                elif model['model_name'] == full_model_name:
                    # Increment count for postfix if base model name matches
                    count += 1
                    full_model_name = f"{base_model_name}__{count}"

        # Add the new model
        new_model = {
            'model_name': full_model_name,
            'litellm_params': {
                'model': f"ollama/{base_model_name}",
                'api_base': api_base
            }
        }
        self.config_data['model_list'].append(new_model)
        self._save_config()

    def get_model_names(self):
        return [model['model_name'] for model in self.config_data['model_list']]

    def remove_model_by_name(self, model_name):
        self.config_data['model_list'] = [model for model in self.config_data['model_list'] if model['model_name'] != model_name]
        self._save_config()

    def remove_model_by_identifier(self, model_identifier, api_base):
        self.config_data['model_list'] = [model for model in self.config_data['model_list'] if not (model['litellm_params']['model'].endswith(model_identifier) and model['litellm_params']['api_base'] == api_base)]
        self._save_config()

    def remove_all_models_by_api_base(self, api_base):
        self.config_data['model_list'] = [model for model in self.config_data['model_list'] if model['litellm_params']['api_base'] != api_base]
        self._save_config()

