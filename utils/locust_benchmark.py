import logging
import gevent.pool
from locust import FastHttpUser, events, task

class MyUser(FastHttpUser):
    data = None

    def on_start(self):
        with open(self.environment.parsed_options.input, "rb") as f:
            self.data = f.read()
        # logging.info("options = %s", self.environment.parsed_options)

    @task
    def my_task(self):
        def concurrent_request(url,data,headers):
            response = self.client.post(
            url=url,
            data=self.data,
            headers=headers
            )
            if response.status_code != 200:
                logging.error("Request failed")
            assert response.status_code == 200
            logging.info(f"response info = {response.json()}")
        
        target_url=f"{self.environment.parsed_options.host}/{self.environment.parsed_options.model_url}"
        logging.info("Request host = %s", target_url)
        headers = {
            "Host": self.environment.parsed_options.custom_header,
            "Content-type": self.environment.parsed_options.content_type,
        }
        pool = gevent.pool.Pool(self.environment.parsed_options.num_users)
        for _ in range(self.environment.parsed_options.num_users):  
            pool.spawn(concurrent_request, target_url,self.data,headers)
        pool.join()


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