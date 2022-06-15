import json

class Config():

    def __init__(self, json_path) -> None:
        data = self.load_json(json_path)

        for key, val in data.items():
            exec(f"self.{key} = val")


    @staticmethod
    def load_json(json_path="config.json"):
        with open(json_path) as json_file:
            data = json.load(json_file)
        
        return data
