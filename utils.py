import yaml
import json

def load_params(params_path:str)-> dict:
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    return params

def save_json(metrics_path:str,metrics_dict:dict):
    with open("metrics.json","w") as f:
        json.dump(metrics_dict,f,indent=4)