import logging
import gevent.pool
from locust import FastHttpUser, events, task ,between

class MyUser(FastHttpUser):
    network_timeout = 500.0  # Timeout for the network operations, in seconds
    connection_timeout = 500.0  # Timeout for establishing a connection, in seconds
    pool_size = 30
    pool = None
    url = ""
    headers = {}

    def on_start(self):
        with open(self.environment.parsed_options.input, "rb") as f:
            self.data = f.read()
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
        with self.client.post(
            url=self.url,
            data=self.data,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"Request failed with status code {response.status_code}")
            else:
                logging.info(f"response info = {response.json()}")
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