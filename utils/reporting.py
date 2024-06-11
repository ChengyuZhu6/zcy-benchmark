import csv
import json
import os
import re
import math

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.common import is_file_empty


def extract_metrics(execution_params, warm_up_lines):
    with open(execution_params["metric_log"]) as f:
        lines = f.readlines()

    click.secho(f"Dropping {warm_up_lines} warmup lines from log", fg="green")
    lines = lines[warm_up_lines:]

    metrics = {
        execution_params["result_file"]: "e2e.txt",
        execution_params["output_log"]: "inference.txt",
    }
    
    all_lines = extract_inference_benchmark_artifacts(execution_params, "inference.txt")

    return metrics


def generate_csv_output(execution_params, metrics, artifacts):
    click.secho("*Generating CSV output...", fg="green")

    # torchserve_artifacts = extract_torchserve_artifacts(execution_params, metrics)

    # artifacts.update(torchserve_artifacts)

    click.secho(f"Saving benchmark results to {execution_params['report_location']}")

    with open(
        os.path.join(execution_params["report_location"], "benchmark", "benchmark_report.csv"),
        "w",
    ) as csv_file:
        csvwriter = csv.writer(csv_file)
        csvwriter.writerow(artifacts.keys())
        csvwriter.writerow(artifacts.values())

    return artifacts


def extract_locust_tool_benchmark_artifacts(execution_params, output_file):
    with open(execution_params["result_file"], "r") as f:
        data = json.load(f)[0]

    response_hist = dict(sorted(data["response_times"].items()))
    keys = [float(k) for k in response_hist.keys()]
    values = [v for v in response_hist.values()]
    all_lines = []
    artifacts = {"Benchmark": "Locust"}
    all_lines.append(f"Request numbers: {data['num_requests']}")
    artifacts["Failed requests"] = data["num_failures"]
    all_lines.append(f"Failed requests: {artifacts['Failed requests']}")
    artifacts["Run time"] = data["total_response_time"] / 1000
    all_lines.append(f"Run time: {artifacts['Run time']}s")
    artifacts["Throughput"] = data["num_requests"] / max(
        data["last_request_timestamp"] - data["start_time"], 0.1
    )
    all_lines.append(f"throughput: {artifacts['Throughput']}")
    artifacts["Latency mean"] = np.multiply(keys, values).sum() / np.sum(values)
    all_lines.append(f"Latency mean: {artifacts['Latency mean']}")
    artifacts["Error rate"] = data["num_failures"] / data["num_requests"] * 100
    all_lines.append(f"Error rate: {artifacts['Error rate']}")
    
    meet_count = 0
    baseline_latency = execution_params['output_length'] * execution_params['average_token_latency']
    print(f"response_hist = {response_hist}")
    for k,v in response_hist.items():
        print(f"k = {k}; v = {v}")
        if int(k) < baseline_latency:
            print(k)
            meet_count = meet_count + v
    all_lines.append(f"Meet Count: {meet_count}")
    out_fname = os.path.join(*(execution_params["tmp_dir"], "benchmark", output_file))
    click.secho(f"\nWriting to {out_fname} ", fg="green")
    with open(out_fname, "w") as outf:
        all_lines = map(lambda x: x + "\n", all_lines)
        outf.writelines(all_lines)
    return artifacts

def extract_inference_benchmark_artifacts(execution_params, output_file):
    latencies = []
    latency_pattern = re.compile(r"'Latency': ([\d\.]+)")
    output_length = execution_params['output_length']
    baseline_latency = float(execution_params['output_length'] * execution_params['average_token_latency']) / 1000
    print(f"baseline_latency: {baseline_latency}")
    meet_count = 0
    with open(execution_params['output_log'], "r") as file:
        for line in file:
            match = latency_pattern.search(line)
            if match:
                latency = float(match.group(1))
                if math.isclose(latency, baseline_latency) or (latency < baseline_latency):
                    print(f"latency: {latency}")
                    meet_count = meet_count + 1
                latencies.append(latency)

    if latencies:
        all_lines = []
        average_latency = sum(latencies) / len(latencies)
        print(f"Average Latency: {average_latency}")
        all_lines.append(f"Average Latency: {average_latency}s")
        all_lines.append(f"Inference Count: {len(latencies)}")
        all_lines.append(f"Meet Count: {meet_count}")
        out_fname = os.path.join(*(execution_params["tmp_dir"], "benchmark", output_file))
        click.secho(f"\nWriting to {out_fname} ", fg="green")
        with open(out_fname, "w") as outf:
            all_lines = map(lambda x: x + "\n", all_lines)
            outf.writelines(all_lines)
    else:
        print("No latency values found in the log file.")


def extract_torchserve_artifacts(execution_params, metrics):
    batched_requests = execution_params["requests"] / execution_params["batch_size"]
    line50 = int(batched_requests / 2)
    line90 = int(batched_requests * 9 / 10)
    line99 = int(batched_requests * 99 / 100)

    artifacts = {}

    with open(
        os.path.join(execution_params["tmp_dir"], "benchmark", "predict.txt")
    ) as f:
        lines = f.readlines()
        lines.sort(key=float)
        artifacts["Model_p50"] = lines[line50].strip()
        artifacts["Model_p90"] = lines[line90].strip()
        artifacts["Model_p99"] = lines[line99].strip()

    with open(
        os.path.join(execution_params["tmp_dir"], "benchmark", "waiting_time.txt")
    ) as f:
        lines = f.readlines()
        lines.sort(key=float)
        num_requests = len(lines)
        line50 = int(num_requests / 2)
        line90 = int(num_requests * 9 / 10)
        line99 = int(num_requests * 99 / 100)
        artifacts["Queue time p50"] = lines[line50].strip()
        artifacts["Queue time p90"] = lines[line90].strip()
        artifacts["Queue time p99"] = lines[line99].strip()

    for m in metrics:
        df = pd.read_csv(
            os.path.join(*(execution_params["tmp_dir"], "benchmark", m)),
            header=None,
            names=["data"],
        )
        if df.empty:
            artifacts[m.split(".txt")[0] + "_mean"] = 0.0
        else:
            artifacts[m.split(".txt")[0] + "_mean"] = df["data"].values.mean().round(2)

    return artifacts


def extract_entity(data, pattern, index, delim=" "):
    pattern = re.compile(pattern)
    for line in data:
        if pattern.search(line):
            return line.split(delim)[index].strip()
    return None


def generate_latency_graph(execution_params):
    click.secho("*Preparing graphs...", fg="green")
    df = pd.read_csv(
        os.path.join(execution_params["tmp_dir"], "benchmark", "predict.txt"),
        header=None,
        names=["latency"],
    )
    iteration = df.index
    latency = df.latency
    a4_dims = (11.7, 8.27)
    plt.figure(figsize=(a4_dims))
    plt.xlabel("Requests")
    plt.ylabel("Prediction time")
    plt.title("Prediction latency")
    plt.bar(iteration, latency)
    plt.savefig(f"{execution_params['report_location']}/benchmark/predict_latency.png")


def generate_profile_graph(execution_params, metrics):
    click.secho("*Preparing Profile graphs...", fg="green")

    plot_data = {}
    for m in metrics:
        file_path = f"{execution_params['tmp_dir']}/benchmark/{m}"
        if is_file_empty(file_path):
            continue
        df = pd.read_csv(file_path, header=None)
        m = m.split(".txt")[0]
        plot_data[f"{m}_index"] = df.index
        plot_data[f"{m}_values"] = df.values

    if execution_params["requests"] > 100:
        sampling = int(execution_params["requests"] / 100)
    else:
        sampling = 1
    click.secho(f"Working with sampling rate of {sampling}")

    a4_dims = (11.7, 8.27)
    grid = plt.GridSpec(3, 2, wspace=0.2, hspace=0.2)
    plt.figure(figsize=a4_dims)
    fig1 = plt.subplot(grid[0, 0])
    fig2 = plt.subplot(grid[0, 1])
    fig3 = plt.subplot(grid[1, 0])
    fig4 = plt.subplot(grid[1, 1])
    fig5 = plt.subplot(grid[2, 0:])

    def plot_line(fig, data, color="blue", title=None):
        fig.set_title(title)
        fig.set_ylabel("Time (ms)")
        fig.set_xlabel("Percentage of queries")
        fig.grid()
        plot_points = np.arange(0, 100, 100 / len(data))
        x = plot_points[: len(data) : sampling]
        y = data[::sampling]
        fig.plot(x, y, f"tab:{color}")

    # Queue Time
    plot_line(
        fig1, data=plot_data["waiting_time_values"], color="pink", title="Queue Time"
    )

    # handler Predict Time
    plot_line(
        fig2,
        data=plot_data["handler_time_values"],
        color="orange",
        title="Handler Time(pre & post processing + inference time)",
    )

    # Worker time
    plot_line(
        fig3,
        data=plot_data["worker_thread_values"],
        color="green",
        title="Worker Thread Time",
    )

    # Predict Time
    plot_line(
        fig4,
        data=plot_data["predict_values"],
        color="red",
        title="Prediction time(handler time+python worker overhead)",
    )

    # Plot in one graph
    plot_line(fig5, data=plot_data["waiting_time_values"], color="pink")
    plot_line(fig5, data=plot_data["handler_time_values"], color="orange")
    plot_line(fig5, data=plot_data["predict_values"], color="red")
    plot_line(
        fig5,
        data=plot_data["worker_thread_values"],
        color="green",
        title="Combined Graph",
    )
    fig5.grid()
    plt.savefig(
        f"{execution_params['report_location']}/benchmark/api-profile1.png",
        bbox_inches="tight",
    )