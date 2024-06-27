import logging
import gevent.pool
from locust import FastHttpUser, events, task ,between
import json
import random
import time
from transformers import LlamaTokenizer

class MyUser(FastHttpUser):
    network_timeout = 500.0  # Timeout for the network operations, in seconds
    connection_timeout = 500.0  # Timeout for establishing a connection, in seconds
    pool_size = 30
    pool = None
    url = ""
    headers = {}
    file_data = ""
    input_data = ""
    model_path = "/media/shared_folder/AIGC-PVC/pvc-91cd0f04-8188-4e39-876e-e9133af61d2f/model-space/Llama-2-7b-hf/"    
    def on_start(self):
        with open(self.environment.parsed_options.input, "r", encoding="utf-8") as f:
            self.input_data = json.load(f)
        self.pool = gevent.pool.Pool(self.pool_size)
        self.url = f"{self.environment.parsed_options.host}/{self.environment.parsed_options.model_url}"
        self.headers = {
            "Host": self.environment.parsed_options.custom_header,
            "Content-type": self.environment.parsed_options.content_type,
        }

    @task
    def my_task(self):
        self.pool.spawn(self.concurrent_request)
        self.pool.join()
        
    def concurrent_request(self):
        random_input = random.choice(self.input_data)
        
        logging.info(f'new_data len: {random_input["input_tokens"]}')
        tic = time.time()
        new_data = {
            "instances": [
                {
                    "text": random_input['text'],
                    "max_tokens": random_input['output_tokens'],
                    "min_tokens": random_input['output_tokens'],
                }
            ]
        }
        with self.client.post(
            url=self.url,
            data=json.dumps(new_data, indent=4),
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Request failed with status code {response.status_code}")
            else:
                toc = time.time()
                logging.info(f'input_tokens: {random_input["input_tokens"]}')
                logging.info(f"E2E start timestamp: {tic}")
                logging.info(f"response info = {response.json()}")
                logging.info(f"E2E end timestamp: {toc}")
                response.success()

@events.init_command_line_parser.add_listener
def init_parser(parser):
    parser.add_argument(
        "--input",
        type=str,
        help="input files",
    )
    parser.add_argument(
        "--content-type",
        type=str,
        help="content type",
    )
    parser.add_argument(
        "--model-url",
        type=str,
        help="model url",
    )
    parser.add_argument(
        "--custom-header",
        type=str,
        help="custom header",
    )