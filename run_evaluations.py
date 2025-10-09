import argparse
import os
import shlex
import subprocess
import sys


# Models to run via OpenRouter using the litellm adapter.
OPENROUTER_MODELS: list[str] = [
    # "openai/gpt-oss-20b:free",
    # "openai/gpt-5-nano",
    # "z-ai/glm-4.6",
    # "anthropic/claude-sonnet-4.5",
    # "openai/gpt-5",
    # "openai/gpt-5-mini",
    # "openai/gpt-5-nano",
    # "openai/gpt-4.1",
    # "openai/gpt-4.1-mini",
    # "google/gemini-2.5-pro",
    "google/gemini-2.5-flash",
    # "anthropic/claude-sonnet-4",
    # "deepseek/deepseek-v3.1-terminus",
    # "qwen/qwen3-next-80b-a3b-thinking",
    # "moonshotai/kimi-k2-0905",
]

# Endpoints to run via a custom inference endpoint (configured in endpoint_model.yaml)
INFERENCE_ENDPOINT_MODELS: list[str] = [
    # "apriel-1.5-15b-thinker",
]

def configure_environment(debug: bool) -> None:
    log_level = "DEBUG" if debug else "INFO"
    os.environ["EVALUATE_IDK_LOG_LEVEL"] = log_level
    os.environ["LITELLM_LOG"] = log_level


def run_command(command: list[str]) -> None:
    print("Running:", shlex.join(command))
    subprocess.run(command, check=True)

def load_tasks_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    # Filter out empty/comment lines
    tasks = [line for line in lines if line and not line.startswith("#")]
    return ",".join(tasks)


tasks_arg = load_tasks_from_file("tasks.txt")

DEFAULT_COMMAND_OPTIONS = [
    tasks_arg,
    "--custom-tasks",
    "custom_tasks.py",
    "--dataset-loading-processes",
    "1",
    "--save-details",
]

def build_litellm_command(
    model_name: str,
    concurrent_requests: int | None,
    max_samples: int | None,
) -> list[str]:
    provider_params = [
        "provider=openrouter",
        f"model_name=openrouter/{model_name}",
        f"concurrent_requests={concurrent_requests}",
    ]

    command: list[str] = [
        "lighteval",
        "endpoint",
        "litellm",
        ",".join(provider_params),
        *DEFAULT_COMMAND_OPTIONS,
    ]
    if max_samples is not None:
        command += ["--max-samples", str(max_samples)]
    return command


def build_inference_endpoint_command(max_samples: int | None) -> list[str]:
    command: list[str] = [
        "lighteval",
        "endpoint",
        "inference-endpoint",
        "endpoint_model.yaml",
        *DEFAULT_COMMAND_OPTIONS,
    ]
    if max_samples is not None:
        command += ["--max-samples", str(max_samples)]
    return command


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Configure and submit evaluation runs.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging and reduce load (max_samples=2, concurrent_requests=1)",
    )
    args = parser.parse_args(argv)

    configure_environment(debug=args.debug)

    # Apply debug overrides only when requested.
    debug_max_samples = 2 if args.debug else None
    debug_concurrent_requests = 1 if args.debug else 10

    for model_name in OPENROUTER_MODELS:
        cmd = build_litellm_command(
            model_name=model_name,
            concurrent_requests=debug_concurrent_requests,
            max_samples=debug_max_samples,
        )
        run_command(cmd)

    for _ in INFERENCE_ENDPOINT_MODELS:
        cmd = build_inference_endpoint_command(max_samples=debug_max_samples)
        run_command(cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


