#!/usr/bin/env python3
"""
Long-running script that submits slurm jobs from a commands file.
Commands file format: commands separated by ==== on its own line.
Each segment between delimiters runs as its own job.
"""

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

RETRY_WAIT_SECONDS = 60

JOB_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=iris
#SBATCH --account=iris
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="24G|48G"
#SBATCH --time=12:00:00
#SBATCH --output=/iris/u/armaana/jobs/logs/%x_%j.out
#SBATCH --error=/iris/u/armaana/jobs/logs/%x_%j.err

. /iris/u/armaana/qc/.venv/bin/activate
cd /iris/u/armaana/qc-pastaware

{commands}
"""


def parse_commands_file(filepath: Path) -> list[str]:
    """Parse commands file, splitting on ==== delimiter."""
    content = filepath.read_text()
    segments = content.split("\n====\n")
    # Filter out empty segments
    return [seg for seg in segments if seg.strip()]


def submit_job(commands: str, template: str) -> bool:
    """
    Attempt to submit a job with the given commands.
    Returns True if successful, False if hit QOS limit.
    Raises exception for other errors.
    """
    job_script = template.format(commands=commands)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(job_script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["sbatch", tmp_path],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"Submitted job: {result.stdout.strip()}")
            return True

        stderr = result.stderr
        if "QOSMaxSubmitJobPerUserLimit" in stderr:
            print(f"Hit QOS limit, will retry...")
            return False

        # Some other error
        raise RuntimeError(f"sbatch failed: {stderr}")

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Submit slurm jobs from a commands file"
    )
    parser.add_argument(
        "commands_file",
        type=Path,
        help="Path to txt file with commands (delimited by ====)",
    )
    parser.add_argument(
        "--template",
        type=Path,
        help="Path to custom job template file (must contain {commands} placeholder)",
    )
    args = parser.parse_args()

    if not args.commands_file.exists():
        raise FileNotFoundError(f"Commands file not found: {args.commands_file}")

    if args.template:
        if not args.template.exists():
            raise FileNotFoundError(f"Template file not found: {args.template}")
        job_template = args.template.read_text()
    else:
        job_template = JOB_TEMPLATE

    command_segments = parse_commands_file(args.commands_file)
    print(f"Found {len(command_segments)} job(s) to submit")

    idx = 0
    while idx < len(command_segments):
        commands = command_segments[idx]
        print(f"\nSubmitting job {idx + 1}/{len(command_segments)}...")

        if submit_job(commands, job_template):
            idx += 1
        else:
            print(f"Waiting {RETRY_WAIT_SECONDS}s before retry...")
            time.sleep(RETRY_WAIT_SECONDS)

    print(f"\nAll {len(command_segments)} jobs submitted successfully!")


if __name__ == "__main__":
    main()
