"""CLI configuration and input handling, outside the original agent loop."""

import argparse
import os
from pathlib import Path

from .config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the original Eyegent agent on an image/query pair.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", type=Path, default=Path("scanpath.png"))
    parser.add_argument("--config", type=Path, help="Optional YAML configuration")
    parser.add_argument("--base-url", help="VLM endpoint including /v1")
    parser.add_argument("--model", help="Model identifier exposed by the server")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Device for CLIPSeg and ScanDDM")
    parser.add_argument("--duration", type=float, help="Original loop's simulated-time threshold")
    parser.add_argument("--max-generations", type=int, help="Original loop's generation counter limit")
    parser.add_argument("--seed", type=int,
                        help="Seed local RNGs and VLM requests; use greedy VLM decoding")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    try:
        config_path = args.config
        if config_path is None and Path("config.yml").is_file():
            config_path = Path("config.yml")
        config = load_config(config_path)
        for name, argument, environment in [
            ("server_url", args.base_url, "EYEGENT_BASE_URL"),
            ("model", args.model, "EYEGENT_MODEL"),
        ]:
            value = argument if argument is not None else os.environ.get(environment)
            if value is not None:
                config["llm"][name] = value
        if "EYEGENT_API_KEY" in os.environ:
            config["llm"]["api_key"] = os.environ["EYEGENT_API_KEY"]
        if args.device:
            config["segmentation"]["device"] = args.device
            config["scanpath"]["device"] = args.device
        if args.duration is not None:
            if args.duration <= 0:
                raise ValueError("--duration must be positive")
            config["inference"]["max_duration"] = args.duration
        if args.max_generations is not None:
            if args.max_generations <= 0:
                raise ValueError("--max-generations must be positive")
            config["inference"]["max_generations"] = args.max_generations
        if not args.image.expanduser().is_file():
            raise FileNotFoundError(args.image)
        if args.output.expanduser().resolve() == args.image.expanduser().resolve():
            raise ValueError("Output must differ from the input image")
        if args.output.suffix.lower() not in {".png", ".pdf", ".svg", ".jpg", ".jpeg"}:
            raise ValueError("Output must have a .png, .pdf, .svg, .jpg, or .jpeg extension")
        if not args.query.strip():
            raise ValueError("Query must not be empty")
        if args.seed is not None:
            if not 0 <= args.seed < 2**32:
                raise ValueError("--seed must be between 0 and 4294967295")
            import random
            import numpy as np
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            import torch
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            config["llm"]["seed"] = args.seed
            config["llm"]["temperature"] = 0.0
        import matplotlib
        matplotlib.use("Agg")
        from .model import Eyegent
        from .plot_scanpaths import save_scanpath
        result = Eyegent(config=config, verbose=args.verbose).run(args.image, args.query)
        output = save_scanpath(args.image, args.query, result, args.output)
        stop = result.stop_reason
        if stop == "max_generations":
            stop += f" ({config['inference']['max_generations']} orchestrator tool calls)"
        print(f"Saved {output} ({len(result.scanpath)} fixations; {stop})")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")
