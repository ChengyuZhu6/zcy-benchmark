import json
import argparse
import os

parser = argparse.ArgumentParser(description="Process some JSON files.")
parser.add_argument("input_file", type=str, help="The JSON input file")

# Parse the arguments
args = parser.parse_args()

# Load questions from 'questions.json'
with open(args.input_file, "r") as file:
    questions_data = json.load(file)
    questions = questions_data["questions"]

data_dir="inputs"
# Create a directory to store the files
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Template for input.json structure
input_template = {"instances": [{"text": ""}]}

# Iterate over each question and create a new JSON file
for i, question in enumerate(questions):
    # Update the 'text' field with the current question
    input_template["instances"][0]["text"] = question

    # Construct the file name
    file_name = f"{data_dir}/input-{i}.json"

    # Write the updated input to a new JSON file
    with open(file_name, "w") as file:
        json.dump(input_template, file, indent=4)
