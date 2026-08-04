from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from threemd.serving import RemoteVLLMServer


def main() -> int:
    parser = argparse.ArgumentParser(description="Start remote vLLM over SSH.")
    parser.add_argument("--host", required=True, help="SSH host or alias.")
    parser.add_argument("--model", required=True, help="Hugging Face model id.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-devices", default=None)
    parser.add_argument("--vllm-bin", default="vllm")
    parser.add_argument("--log-path", default="vllm-3md.log")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    extra_args = tuple(arg for arg in args.extra_args if arg != "--")
    server = RemoteVLLMServer(
        host=args.host,
        model=args.model,
        port=args.port,
        gpu_devices=args.gpu_devices,
        vllm_bin=args.vllm_bin,
        log_path=args.log_path,
        extra_args=extra_args,
    )
    command = server.start_command()
    print(" ".join(shlex.quote(part) for part in command))
    if args.dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
