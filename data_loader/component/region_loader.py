import json

def load_json_region(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data