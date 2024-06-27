import json
import random
import time
from transformers import LlamaTokenizer    
import logging

def get_format_string(model_name):
        """Get the format string."""
        known_system_prompts = {
            "llama": "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]",
            "flan": "Question: {prompt}\n\nAnswer:",
            "gpt-neox": "{system_prompt}\n\n{prompt}",
            "starcoder": "input: \n\n{prompt} \n\noutput:",
        }

        for name, fmt_str in known_system_prompts.items():
            if name in model_name:
                logging.info("Using %s prompt format, model_name: %s", name, model_name)
                return fmt_str

        logging.info("Using default prompt format model_name: %s", model_name)
        return "{system_prompt}\n\n{prompt}"

model_path = "/media/shared_folder/AIGC-PVC/pvc-91cd0f04-8188-4e39-876e-e9133af61d2f/model-space/Llama-2-7b-hf/"
tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
prompt_format = get_format_string("llama")
input_data = []
input_file_path = "./openorca_large_subset_011.jsonl"
prompt_token_limits = [128,256,512,1024,2048]
output_token_limits = [128]
constant_input_datas = {f"{input_id}_{output_id}": [] for input_id in prompt_token_limits for output_id in output_token_limits}
with open(input_file_path, "r", encoding="utf-8") as f:
    file_data = f.readlines()[1:]
    for line in file_data:
        json_line = json.loads(line.strip())
        input_prompt = prompt_format.format(prompt=json_line["question"], system_prompt=json_line["system_prompt"])
        prompt_token_ids = tokenizer(input_prompt).input_ids
        for input_limit in prompt_token_limits:
            for output_limit in output_token_limits:
                if len(prompt_token_ids) >= input_limit:
                    new_data_token_ids = prompt_token_ids[:input_limit]
                    new_data_decoded_text = tokenizer.decode(new_data_token_ids, skip_special_tokens=True)
                    constant_input_datas[f"{input_limit}_{output_limit}"].append({
                        "text": new_data_decoded_text,
                        "input_tokens": input_limit,
                        "output_tokens": output_limit,
                    })

for input_limit in prompt_token_limits:
    for output_limit in output_token_limits:
        # Write the data to the JSON file
        output_file_path = f"{input_limit}_{output_limit}.json"
        with open(output_file_path, 'w') as json_file:
            json.dump(constant_input_datas[f"{input_limit}_{output_limit}"], json_file, indent=4)