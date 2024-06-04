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

        amp_dtype = getattr(torch, "bfloat16")
        self.tokenizer = LlamaTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, low_cpu_mem_usage=True).to(self.device)
        model.config.use_cache = True
        model.config.lm_head_generation = True
        model = model.to(memory_format=torch.channels_last)

        self.model = ipex.llm.optimize(
            model.eval(),
            dtype=amp_dtype,
            inplace=True,
            deployment_mode=False,
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
        input_ids = self.tokenizer(prompt, return_tensors='pt')

        logging.info("llm inference get inputs")
        logging.info(input_ids)
        first_token_latency=0
        is_first_token=False
        all_token_latency=0
        skip_special_tokens=True
        clean_up_tokenization_spaces=False
        spaces_between_special_tokens=True
        with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
            enabled=True
        ):
            logging.info("llm inference get output_ids")
            output_ids = self.model.generate(**input_ids, max_length = self.max_length)
            filtered_tokens=[]
            for output_id in output_ids:
                added_vocab = self.tokenizer.get_added_vocab()
                for out_id in output_id:
                    start_time = time.perf_counter()
                    id_to_token=self.tokenizer.convert_ids_to_tokens([out_id], skip_special_tokens=skip_special_tokens)
                    logging.info(id_to_token)
                    if len(id_to_token) > 0:
                        filtered_tokens.append(id_to_token[0])
                    end_time = time.perf_counter()
                    time_cost_milliseconds = (end_time - start_time) * 1000
                    if is_first_token == False:
                        first_token_latency = time_cost_milliseconds
                        is_first_token=True
                    all_token_latency = all_token_latency + time_cost_milliseconds
                    print(f"Single token {time_cost_milliseconds:.2f} milliseconds.")
                # filtered_tokens = self.tokenizer.convert_ids_to_tokens(output_id, skip_special_tokens=skip_special_tokens)
                legacy_added_tokens = set(added_vocab.keys()) - set(self.tokenizer.all_special_tokens) | {
                    token for token in self.tokenizer.additional_special_tokens if self.tokenizer.convert_tokens_to_ids(token) >= self.tokenizer.vocab_size
                }
                
                sub_texts = []
                current_sub_text = []
                for token in filtered_tokens:
                    if skip_special_tokens and token in self.tokenizer.all_special_ids:
                        continue
                    
                    if token in legacy_added_tokens:
                        if current_sub_text:
                            string = self.tokenizer.convert_tokens_to_string(current_sub_text)
                            if len(string) > 0:
                                sub_texts.append(string)
                            current_sub_text = []
                        sub_texts.append(token)
                    else:
                        current_sub_text.append(token)
                if current_sub_text:
                    sub_texts.append(self.tokenizer.convert_tokens_to_string(current_sub_text))
                if spaces_between_special_tokens:
                    text = " ".join(sub_texts)
                else:
                    text = "".join(sub_texts)
                    
                average_latency = float(all_token_latency)/float(len(filtered_tokens))
                logging.info("all tokens llm inference result:")
                logging.info(text)
                logging.info(f"first_token_latency {first_token_latency:.2f}")
                logging.info(f"average_latency {average_latency:.2f}")
        return {
            "output": text,
            "first_token_latency": first_token_latency,
            "average_latency": average_latency,
        }

    def postprocess(self, data):
        return [data]
