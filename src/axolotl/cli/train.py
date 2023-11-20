"""
CLI to run training on a model
"""
import logging
from pathlib import Path
import os
import sys
import fire
import tempfile
import transformers
import time
import json
from pathlib import Path
from colorama import Fore
import requests
from axolotl.cli import (
    check_accelerate_default_config,
    check_user_token,
    load_cfg,
    load_datasets,
    print_axolotl_text_art,
)
from axolotl.common.cli import TrainerCliArgs
from axolotl.common.const import DEFAULT_DATASET_PREPARED_PATH
from axolotl.train import train

LOG = logging.getLogger("axolotl.cli.train")


def do_cli(config: Path = Path("examples/"), **kwargs):
    getJobURL = os.environ.get("HELIX_NEXT_TASK_URL", None)
    
    if getJobURL is None:
        sys.exit("HELIX_GET_JOB_URL is not set")

    while True:
        response = requests.get(getJobURL)
        if response.status_code != 200:
            time.sleep(0.1)
            continue

        task = json.loads(response.content)

        print("🟣 Mistral Job --------------------------------------------------\n")
        print(task)

        session_id = task["session_id"]
        finetune_file = task["finetune_file"]

        results_dir = f"/tmp/helix/results/{session_id}"
        
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        # pylint: disable=duplicate-code
        print_axolotl_text_art()
        parsed_cfg = load_cfg(config, **kwargs)

        parsed_cfg["datasets"][0]["path"] = finetune_file
        parsed_cfg["datasets"][0]["type"] = "sharegpt"
        parsed_cfg["datasets"][0]["ds_type"] = "json"
        parsed_cfg["output_dir"] = results_dir

        import pprint; pprint.pprint(parsed_cfg)

        check_accelerate_default_config()
        check_user_token()
        parser = transformers.HfArgumentParser((TrainerCliArgs))
        parsed_cli_args, _ = parser.parse_args_into_dataclasses(
            return_remaining_strings=True
        )
        print(f"[SESSION_START]session_id={session_id}", file=sys.stdout)
        dataset_meta = load_datasets(cfg=parsed_cfg, cli_args=parsed_cli_args)
        train(cfg=parsed_cfg, cli_args=parsed_cli_args, dataset_meta=dataset_meta)
        print(f"[SESSION_END]results_dir={results_dir}", file=sys.stdout)


if __name__ == "__main__":
    fire.Fire(do_cli)
