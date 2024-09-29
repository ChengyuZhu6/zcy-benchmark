import io
import os
import logging
import torch
from typing import Dict, List, Union

from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer
import json
import time
import intel_extension_for_pytorch as ipex
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class LlmClassifier(BaseHandler):
    def __init__(self):
        self.initialized = False
        self.max_length = 256

    def initialize(self, ctx):
        properties = ctx.system_properties
        model_dir = "/mnt/models/models/"
        amp_dtype = getattr(torch, "bfloat16")

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        logger.info("Initialize: Start to load model")

        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_dir, trust_remote_code=True
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            .eval()
            .to(self.device)
        )
        logger.info("Initialize: model loaded")

    def preprocess(self, requests):
        logger.info("Preprocess start, requests:")
        logger.info(requests)

        for _, req in enumerate(requests):
            payload = req["body"]
            # protocol v2
            if "inputs" in payload and isinstance(payload["inputs"], list):
                req_inputs = payload.get("inputs")
            elif (
                isinstance(payload, Dict)
                and "instances" in payload
                and isinstance(payload["instances"], list)
            ):
                req_inputs = payload.get("instances")

            for _, row in enumerate(req_inputs):
                question = row.get("text")

        logger.info("llm preprocess done")
        logger.info(question)
        return question

    def inference(self, inputs):
        prompt = inputs
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        logging.info("llm inference get inputs")
        logging.info(input_ids)
        output_ids = []
        first_token_latency=0
        is_first_token=False
        all_token_latency=0
        for _ in range(self.max_length):
            start_time = time.perf_counter()
            with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
                enabled=True
            ):
                # output_ids = self.model.generate(**inputs)
                logging.info("llm inference get output_ids")
                output = self.model(input_ids)

            next_token_logits = output.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            output_ids.append(next_token.item())
            logging.info("llm inference get next_token")
            logging.info(next_token)
            next_token_output = self.tokenizer.decode([next_token.item()])
            logging.info(next_token_output)
            end_time = time.perf_counter()
            time_cost_milliseconds = (end_time - start_time) * 1000
            if is_first_token == False:
                first_token_latency = time_cost_milliseconds
                is_first_token=True
            all_token_latency = all_token_latency + time_cost_milliseconds
            print(f"Single token {time_cost_milliseconds:.2f} milliseconds.")
            # Update the input_ids to include the new token
            input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
            
            # Check if the next_token is the end of sentence token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        decoded_output = self.tokenizer.decode(output_ids)
        average_latency = float(all_token_latency)/float(len(output_ids))
        logging.info("all tokens llm inference result:")
        logging.info(decoded_output)
        logging.info(f"first_token_latency {first_token_latency:.2f}")
        logging.info(f"average_latency {average_latency:.2f}")
        return {
            "output": decoded_output,
            "first_token_latency": first_token_latency,
            "average_latency": average_latency,
        }

    def postprocess(self, data):
        return [data]
