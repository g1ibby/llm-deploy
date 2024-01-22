import yaml
import os

class LLMsConfig:
    def __init__(self, filename="llms.yaml"):
        self.data = {'models': {}}
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                self.data = yaml.safe_load(file) or {'models': {}}

    def get_models(self):
        models = []
        for name, details in self.data['models'].items():
            # Check if both 'model' and 'priority' are present
            if 'model' in details and 'priority' in details:
                # Check if 'priority' is either 'high' or 'low'
                if details['priority'] in ['high', 'low']:
                    models.append({
                        'name': name,
                        'model': details['model'],
                        'priority': details['priority']
                    })
                else:
                    raise ValueError(f"Invalid priority value for {name}: {details['priority']}")
            else:
                raise KeyError(f"Missing required keys in model {name}")
        return models

