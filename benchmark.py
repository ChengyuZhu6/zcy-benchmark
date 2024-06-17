import json
import os
import tempfile

import click
import click_config_file
from utils.benchmarks import create_benchmark

def json_provider(file_path, cmd_name):
    with open(file_path) as config_data:
        return json.load(config_data)


@click.command()
@click.argument("test_plan", default="custom")
@click.option(
    "--url",
    "-u",
    default="https://torchserve.pytorch.org/mar_files/resnet-18.mar",
    help="Input model url",
)
@click.option(
    "--concurrency", "-c", default=10, help="Number of concurrent requests to run"
)
@click.option("--requests", "-r", default=100, help="Number of requests")
@click.option(
    "--input",
    "-i",
    default="../examples/image_classifier/kitten.jpg",
    type=click.Path(exists=True),
    help="The input file path for model",
)
@click.option(
    "--content_type", "-ic", default="application/jpg", help="Input file content type"
)
@click.option(
    "--generate_graphs",
    "-gg",
    default=False,
    help="Enable generation of Graph plots. Default False",
)
@click.option(
    "--inference_model_url",
    "-imu",
    default="predictions/benchmark",
    help="Inference function url - can be either for predictions or explanations. Default predictions/benchmark",
)
@click.option(
    "--report_location",
    "-rl",
    default=tempfile.gettempdir(),
    help=f"Target location of benchmark report. Default {tempfile.gettempdir()}",
)
@click.option(
    "--tmp_dir",
    "-td",
    default=tempfile.gettempdir(),
    help=f"Location for temporal files. Default {tempfile.gettempdir()}",
)
@click.option(
    "--benchmark_backend",
    "-bb",
    type=click.Choice(["locust"], case_sensitive=False),
    default="locust",
    help=f"Benchmark backend to use.",
)
@click.option(
    "--output_length",
    "-ol",
    default=128,
    help=f"the length of output tokens.",
)
@click.option(
    "--average_token_latency",
    "-atl",
    default=120,
    help=f"the average latency of generating token (ms).",
)
@click.option(
    "--custom_header",
    "-ch",
    default="",
    help=f"Custom header to a request.",
)
@click.option(
    "--run_time",
    "-rt",
    default="60s",
    help=f"run time.",
)
@click.option(
    "--requests",
    "-r",
    default=0,
    help=f"requests",
)
@click_config_file.configuration_option(
    provider=json_provider, implicit=False, help="Read configuration from a JSON file"
)

def benchmark(test_plan, **input_params):
    execution_params = input_params.copy()

    # set params
    update_exec_params(execution_params, input_params)

    click.secho("Starting benchmark suite...", fg="green")
    click.secho("\n\nConfigured execution parameters are:", fg="green")
    click.secho(f"{execution_params}", fg="blue")

    benchmark = create_benchmark(execution_params)

    benchmark.prepare_environment()

    # benchmark.warm_up()
    benchmark.run()

    click.secho("Bench Execution completed.", fg="green")

    benchmark.generate_report()
    click.secho("\nTest suite execution complete.", fg="green")


def update_exec_params(execution_params, input_param):
    execution_params.update(input_param)

    execution_params["result_file"] = os.path.join(
        execution_params["tmp_dir"], "benchmark", "result.txt"
    )
    execution_params["metric_log"] = os.path.join(
        execution_params["tmp_dir"], "benchmark", "logs", "model_metrics.log"
    )
    execution_params["output_log"] = os.path.join(
        execution_params["tmp_dir"], "benchmark", "logs", "output_log.log"
    )


if __name__ == "__main__":
    benchmark()