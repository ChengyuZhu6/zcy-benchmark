import io
import os
import logging
import torch
import numpy as np
from itertools import chain
from typing import Dict, List, Union
from transformers import AutoModelForCausalLM, LlamaTokenizer, TextStreamer
import json
import time
import intel_extension_for_pytorch as ipex
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class LlmClassifier(BaseHandler):
    def __init__(self):
        self.initialized = False
        self.max_length = 1024
        self.max_new_tokens = 128
        self.streamer = None
        self.num_beams = 1
        self.lm_head_generation = True
        self.use_cache = False
        # self.batch_size = 1
        self.token_latency = True

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

        self.tokenizer = LlamaTokenizer.from_pretrained(model_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        model.config.lm_head_generation = self.lm_head_generation
        model.config.text_max_length = self.max_length
        model.config.max_seq_len = self.max_new_tokens
        model.config.use_cache = self.use_cache
        # model.config.batch_size = self.batch_size
        model.config.token_latency = self.token_latency
        model = model.to(memory_format=torch.channels_last)

        self.model = ipex.llm.optimize(
            model.eval(),
            dtype=amp_dtype,
            inplace=True,
            deployment_mode=False,
        )

        logger.info("Initialize: model loaded")

    def preprocess(self, requests):
        questions = []
        max_new_token_length = self.max_new_tokens
        logger.info("Preprocess start, requests:")
        logger.info(f"requests size = {len(requests)}")
        for _, req in enumerate(requests):
            payload = req["body"]
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
                questions.append(question)
                max_new_token_len = row.get("max_new_tokens")
                if max_new_token_len != None:
                    max_new_token_length = max_new_token_len

        logger.info("llm preprocess done")
        return {
            "prompts": questions,
            "max_new_tokens": max_new_token_length,
        }

    def inference(self, inputs):
        max_new_token_length = inputs["max_new_tokens"]
        prompts = inputs["prompts"]
        generate_kwargs = dict(
            do_sample=False,
            temperature=0.9,
            num_beams=self.num_beams,
            max_new_tokens=max_new_token_length,
            min_new_tokens=max_new_token_length,
            streamer=self.streamer,
        )

        tic = time.time()

        with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(enabled=True):
            input_ids = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)
            logging.info("llm inference get inputs")

            outputs = self.model.generate(input_ids, **generate_kwargs)
            logging.info(f"Output length/shape: {outputs.shape}")
            logging.info(f"Outputs: {outputs}")
            
            truncated_outputs = outputs[:, -max_new_token_length:]
            logging.info(f"truncated_outputs length/shape: {truncated_outputs.shape}")
            logging.info(f"truncated_outputs: {truncated_outputs}")            
            gen_texts = self.tokenizer.batch_decode(truncated_outputs, skip_special_tokens=True)
            toc = time.time()

            total_time = toc - tic

            print("Inference latency: %.3f sec." % total_time)

        return {
            "Output": gen_texts,
            "Latency": total_time,
        }

    def postprocess(self, data):
        outputs = data["Output"]
        latency = data["Latency"]
        response_list = [{"Output": output, "Latency": latency} for output in outputs]
        logger.info(f"Postprocess: generated {len(response_list)} responses")
        return response_list
