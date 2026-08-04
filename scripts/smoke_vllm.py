from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from urllib import request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from threemd.prompts import PromptBuilder
from threemd.responses import has_answer_line, strip_gemma_thought_channel
from threemd.serving import RemoteVLLMServer


def post_json(url: str, payload: dict, api_key: str) -> dict:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    req = request.Request(url, data=data, headers=headers, method="POST")
    with request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test vLLM chat completions.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-root", default="02_CODE/configs/prompts")
    parser.add_argument("--arm", default="BASE")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--no-system-role", action="store_true")
    args = parser.parse_args()

    health_url = RemoteVLLMServer.health_url(args.base_url)
    with request.urlopen(health_url, timeout=10) as response:
        print(response.read().decode("utf-8"))

    messages = PromptBuilder(
        args.prompt_root,
        use_system_role=not args.no_system_role,
    ).build(
        arm=args.arm,
        question=(
            "A 22-year-old college student has fever, headache, neck stiffness, "
            "and photophobia. Cerebrospinal fluid Gram stain shows gram-negative "
            "diplococci. Which organism is the most likely cause?\n\n"
            "Options:\n"
            "A. Streptococcus pneumoniae\n"
            "B. Neisseria meningitidis\n"
            "C. Listeria monocytogenes\n"
            "D. Haemophilus influenzae"
        ),
        answer_spec="A, B, C, or D",
    )
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "max_completion_tokens": 1024,
    }
    result = post_json(f"{args.base_url.rstrip('/')}/chat/completions", payload, args.api_key)
    raw_content = result["choices"][0]["message"]["content"]
    content = strip_gemma_thought_channel(raw_content)
    print(content)
    if not has_answer_line(content):
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
