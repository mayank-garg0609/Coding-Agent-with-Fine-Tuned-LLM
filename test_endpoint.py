import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
endpoint = os.environ.get("GEMMA_ENDPOINT")

payload = {
    "prompt": "<start_of_turn>user\nYou are an autonomous coding agent capable of planning and executing tasks using tools.\nAvailable tools:\n1. read_file\n2. write_file\n3. list_directory\n4. execute_command\n\nCreate a simple FastAPI app in a file named main.py that has one endpoint /hello which returns {'message': 'hello world'}.<end_of_turn>\n<start_of_turn>model\n",
    "max_length": 512
}

resp = requests.post(endpoint, json=payload)
print(resp.json())
