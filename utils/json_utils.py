import json

def save_json(data, output_path):
    """
    Save data to a JSON file.
    """
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(input_path):
    """
    Load data from a JSON file.
    """
    with open(input_path, 'r') as f:
        return json.load(f)
