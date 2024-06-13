import io
import os
import logging
import torch
import numpy as np
from itertools import chain
from typing import Dict, List, Union

from transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer
from transformers import TextStreamer

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
        self.use_cache = True
        self.batch_size = 1
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

        amp_dtype = getattr(torch, "bfloat16")
        self.tokenizer = LlamaTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        model.config.lm_head_generation = self.lm_head_generation
        model.config.text_max_length = self.max_length
        model.config.max_seq_len = self.max_new_tokens
        model.config.use_cache = self.use_cache
        model.config.batch_size = self.batch_size
        model.config.token_latency = self.token_latency
        model = model.to(memory_format=torch.channels_last)

        self.model = ipex.llm.optimize(
            model.eval(),
            dtype=amp_dtype,
            inplace=True,
            deployment_mode=False,
        )

        self.streamer = TextStreamer(self.tokenizer)

        logger.info("Initialize: model loaded")

    def preprocess(self, requests):
        logger.info("Preprocess start, requests:")
        # logger.info(requests)

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
        # logger.info(question)
        return question

    def inference(self, inputs):
        prompt = inputs
        total_list = []
        generate_kwargs = dict(
            do_sample=False,
            temperature=0.9,
            num_beams=self.num_beams,
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.max_new_tokens,
            streamer=self.streamer,
        )

        tic = time.time()

        with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
            enabled=True
        ):

            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            logging.info("llm inference get inputs")
            # logging.info(input_ids)

            output = self.model.generate(input_ids, **generate_kwargs)
            # logging.info(f"output: {output}")
            logging.info(
                f"Output length/shape: {len(output) if isinstance(output, list) else output.shape}"
            )
            gen_ids = output[0] if self.token_latency else output
            gen_text = self.tokenizer.batch_decode(
                gen_ids,
                skip_special_tokens=True,
            )
            toc = time.time()
            input_tokens_lengths = [x.shape[0] for x in input_ids]
            output_tokens_lengths = (
                [gen_ids.shape[0]]
                if self.token_latency
                else [x.shape[0] for x in output]
            )
            total_new_tokens = [
                o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)
            ]
            # print(gen_text, total_new_tokens, flush=True)
            print("Time: %.6f sec" % (toc - tic), flush=True)
            total_time = toc - tic
            # if self.token_latency:
            #     total_list.append(output[1])
        print("Inference latency: %.3f sec." % total_time)
        # if self.token_latency:
        #     first_latency = np.mean([x[0] for x in total_list])
        #     average_2n = list(chain(*[x[1:] for x in total_list]))
        #     average_2n.sort()
        #     average_2n_latency = np.mean(average_2n)
        #     p90_latency = average_2n[int(len(average_2n) * 0.9)]
        #     p99_latency = average_2n[int(len(average_2n) * 0.99)]
        #     print("First token latency: %.3f sec." % first_latency)
        #     print("Average 2... latency: %.3f sec." % average_2n_latency)
        #     print("P90 2... latency: %.3f sec." % p90_latency)
        #     print("P99 2... latency: %.3f sec." % p99_latency)
        #     logging.info("all tokens llm inference result:")
        #     logging.info(gen_text)
        #     logging.info(f"text length {len(gen_text)}")
        # if self.token_latency:
        #     return {
        #         "Output": gen_text,
        #         "First token latency": first_latency,
        #         "Average latency": average_2n_latency,
        #         "P90 latency": p90_latency,
        #         "P99 latency": p99_latency,
        #     }
        # else:
        #     return {
        #         "Output": gen_text,
        #         "Latency": total_time,
        #     }
        return {
            "Output": total_new_tokens,
            "Latency": total_time,
        }

    def postprocess(self, data):
        return [data]
