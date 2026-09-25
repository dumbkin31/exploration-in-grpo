#!/usr/bin/env python
"""Rollout throughput and peak GPU memory on ONE 2080 Ti (run inside an allocation).

Answers the two numbers the memory/throughput plan depends on before any real training run:

    tokens/s of vLLM (fp16, TRITON_ATTN on sm_75) at several concurrencies, and
    peak GPU memory used by the engine (polled with nvidia-smi, since the engine core runs in
    its own process).

Prompts come from the smoke parquet (real MATH problems, chat template, non-thinking) or a
synthetic prompt when no data is staged. Results go to stdout and ``--out`` as JSON.

    python scripts/bench_rollout.py --model $MC_MODEL_DIR --concurrency 1 4 8 16 --max-tokens 1024
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from mc_data.schema import build_messages  # noqa: E402

SYNTHETIC = [
    "Find the sum of all positive integers n such that n^2 + 2n + 3 is divisible by 7 and n < 100.",
    "Let f(x) = x^3 - 3x + 1. How many real roots does f have? Justify each step.",
    "Compute the number of ordered pairs (a, b) of positive integers with a + b = 40 and gcd(a, b) = 4.",
    "A fair die is rolled 5 times. What is the probability that the product of the rolls is even?",
]


class GpuMemPoller(threading.Thread):
    """Samples nvidia-smi memory.used for every GPU until stopped; keeps the per-GPU max."""

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_mib: list[int] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout
                mems = [int(float(line.split(",")[1])) for line in out.strip().splitlines() if "," in line]
                if len(mems) > len(self.peak_mib):
                    self.peak_mib.extend([0] * (len(mems) - len(self.peak_mib)))
                self.peak_mib = [
                    max(a, b)
                    for a, b in zip(self.peak_mib, mems + [0] * (len(self.peak_mib) - len(mems)), strict=True)
                ]
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval)

    def stop(self) -> list[int]:
        self._stop.set()
        self.join(timeout=5)
        return self.peak_mib


def load_prompts(data_dir: str | None, n: int) -> list[list[dict]]:
    convs: list[list[dict]] = []
    if data_dir:
        p = Path(data_dir) / "smoke_train.parquet"
        if p.exists():
            import pandas as pd

            df = pd.read_parquet(p)
            convs = [list(x) for x in df["prompt"]]
    if not convs:
        convs = [build_messages(q) for q in SYNTHETIC]
    while len(convs) < n:
        convs = convs + convs
    return convs[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("MC_MODEL_DIR"))
    ap.add_argument("--data-dir", default=os.environ.get("MC_DATA_DIR"))
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size (4 = the training layout)")
    ap.add_argument(
        "--attention-backend", default="TRITON_ATTN", help="vLLM attention backend (sm_75: TRITON_ATTN)"
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.model:
        print("ERROR: --model or MC_MODEL_DIR required", file=sys.stderr)
        return 2

    import torch
    from vllm import LLM, SamplingParams

    info = {
        "model": args.model,
        "gpu": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vllm": __import__("vllm").__version__,
        "attention_backend": args.attention_backend,
        "tensor_parallel_size": args.tp,
        "max_tokens": args.max_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    print(json.dumps(info, indent=2, default=str))

    poller = GpuMemPoller()
    poller.start()
    t0 = time.time()
    llm = LLM(
        model=args.model,
        dtype="float16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tp,
        attention_config={"backend": args.attention_backend},
        seed=0,
    )
    info["engine_startup_seconds"] = time.time() - t0
    info["peak_mib_after_startup"] = poller.peak_mib

    results = []
    for conc in args.concurrency:
        convs = load_prompts(args.data_dir, conc)
        sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, ignore_eos=True, seed=0)
        # warm-up (cuda graphs / triton autotune) then timed run
        llm.chat(
            convs[:1],
            sampling_params=SamplingParams(temperature=1.0, max_tokens=32),
            use_tqdm=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        t0 = time.time()
        outs = llm.chat(
            convs, sampling_params=sp, use_tqdm=False, chat_template_kwargs={"enable_thinking": False}
        )
        dt = time.time() - t0
        gen_tokens = sum(len(o.token_ids) for req in outs for o in req.outputs)
        prompt_tokens = sum(len(req.prompt_token_ids) for req in outs)
        row = {
            "concurrency": conc,
            "seconds": round(dt, 2),
            "generated_tokens": gen_tokens,
            "prompt_tokens": prompt_tokens,
            "gen_tokens_per_s": round(gen_tokens / dt, 1),
            "per_seq_tokens_per_s": round(gen_tokens / dt / conc, 1),
            "peak_mib_so_far": list(poller.peak_mib),
        }
        print(json.dumps(row))
        results.append(row)

    info["peak_mib"] = poller.stop()
    info["results"] = results
    kv_per_token_kib = 112  # Qwen3-1.7B: 2 x 28 layers x 8 kv heads x 128 dims x 2 bytes (whole model)
    info["note"] = (
        f"KV cache is ~{kv_per_token_kib} KiB/token across all TP ranks; with gpu_memory_utilization="
        f"{args.gpu_memory_utilization} and tp={args.tp} the engine log line 'GPU KV cache size: N tokens' gives "
        "the concurrency budget at your max_model_len. Grep the log for 'Using ... attention backend' too."
    )
    print(json.dumps(info, indent=2, default=str))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(info, indent=2, default=str))
        print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
