"""
Generate synthetic LoRA fine-tuning data using Qwen 2.5 7B via Ollama.

For each abstract, Qwen extracts structured JSON in our schema.
Output: data/lora_train.jsonl — 1000 (abstract, JSON) training pairs.

Usage:
    python scripts/synthesize_data.py
    python scripts/synthesize_data.py --n_samples 1000 --output data/lora_train.jsonl
"""
import argparse
import json
import random
import time
import re
from pathlib import Path

import requests
from datasets import load_dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"

SCHEMA_KEYS = [
    "main_contribution",
    "methods",
    "datasets_evaluated",
    "baselines_compared",
    "claimed_improvement",
    "task_domain",
]

PROMPT_TEMPLATE = """Extract structured information from this machine learning abstract. Return ONLY a valid JSON object with exactly these keys:
- main_contribution (string): the primary novelty or contribution
- methods (list of strings): techniques, models, or algorithms proposed or used
- datasets_evaluated (list of strings): datasets used for experiments
- baselines_compared (list of strings): existing methods compared against
- claimed_improvement (string): quantitative or qualitative improvement claimed
- task_domain (string): the ML task and domain (e.g. "Image Classification / Computer Vision")

If a field cannot be determined from the abstract, use an empty string or empty list.
Do not include any explanation, markdown, or text outside the JSON object.

Abstract:
{abstract}

JSON:"""


def call_ollama(prompt: str, timeout: int = 30) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def extract_json(raw: str) -> dict | None:
    """Try to extract valid JSON from model output."""
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object within the text
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def validate_schema(obj: dict) -> dict:
    """Ensure all required keys exist with correct types."""
    result = {}
    for key in SCHEMA_KEYS:
        val = obj.get(key, "" if key in ("main_contribution", "claimed_improvement", "task_domain") else [])
        # Coerce lists if model returned a string
        if key in ("methods", "datasets_evaluated", "baselines_compared"):
            if isinstance(val, str):
                val = [val] if val else []
        # Coerce strings if model returned a list
        elif isinstance(val, list):
            val = ", ".join(val) if val else ""
        result[key] = val
    return result


def synthesize_one(abstract: str, max_retries: int = 3) -> dict | None:
    prompt = PROMPT_TEMPLATE.format(abstract=abstract.strip())
    for attempt in range(max_retries):
        try:
            raw = call_ollama(prompt)
            obj = extract_json(raw)
            if obj:
                return validate_schema(obj)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"    Failed after {max_retries} attempts: {e}")
        time.sleep(0.5)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--output", default="data/lora_train.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Skip already processed abstracts")
    args = parser.parse_args()

    random.seed(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading dataset...")
    ds = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
    all_texts = []
    for ex in ds:
        title = ex.get("title", "") or ""
        abstract = ex.get("abstract", "") or ""
        text = (title + "\n" + abstract).strip()
        if len(text) > 100:  # skip very short entries
            all_texts.append(text)

    print(f"Total usable abstracts: {len(all_texts):,}")

    # Sample n_samples
    sampled = random.sample(all_texts, min(args.n_samples, len(all_texts)))
    print(f"Sampled {len(sampled):,} abstracts for synthesis")

    # Resume: load already processed abstracts
    already_done = set()
    if args.resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    already_done.add(entry["input"][:100])
                except:
                    pass
        print(f"Resuming — {len(already_done)} already done, skipping...")

    # Run synthesis
    success, failed = 0, 0
    t0 = time.time()

    with open(out_path, "a" if args.resume else "w") as f:
        for i, text in enumerate(sampled):
            if text[:100] in already_done:
                continue

            result = synthesize_one(text)

            if result:
                entry = {"input": text, "output": result}
                f.write(json.dumps(entry) + "\n")
                f.flush()
                success += 1
            else:
                failed += 1

            # Progress log every 10
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                remaining = (len(sampled) - i - 1) / rate
                print(f"  [{i+1}/{len(sampled)}] success={success} failed={failed} "
                      f"| {rate:.1f}/min | ETA {remaining/60:.1f}min")

    print(f"\nDone. {success} examples saved to {out_path}")
    print(f"Failed: {failed} ({failed/len(sampled)*100:.1f}%)")
    print(f"Total time: {(time.time()-t0)/60:.1f} minutes")


if __name__ == "__main__":
    main()
