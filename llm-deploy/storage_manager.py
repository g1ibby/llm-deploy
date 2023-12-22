import json
from pathlib import Path

class StorageManager:
    def __init__(self, filename="state.json"):
        self.filename = filename
        self.data = self._load_data()

    def _load_data(self):
        """ Load data from a JSON file. """
        if Path(self.filename).exists():
            with open(self.filename, 'r') as file:
                return json.load(file)
        else:
            return {}

    def _save_data(self):
        """ Save data to a JSON file. """
        with open(self.filename, 'w') as file:
            json.dump(self.data, file, indent=4)

    def sync_instances(self, ids):
        # Convert incoming IDs to strings
        string_ids = set(str(id) for id in ids)
        # Remove instances not in the new list of IDs
        ids_to_remove = set(self.data.keys()) - string_ids
        for id in ids_to_remove:
            del self.data[id]

        for id in string_ids:
            if id not in self.data:
                self.data[id] = {"ollama_addr": ""}

        # Save changes
        self._save_data()

    def save_instance(self, id, value):
        """ Add or update a record in the data. """
        self.data[str(id)] = value
        self._save_data()

    def get_instance(self, id):
        """ Retrieve a single record from the data. """
        return self.data.get(str(id), None)
