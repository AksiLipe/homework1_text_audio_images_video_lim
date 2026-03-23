import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Local open-source LLM (will be downloaded from Hugging Face on first run).",
    )
    parser.add_argument("--prompt", default="Привет! Объясни, что такое ML и DL простыми словами.")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_kwargs = {"low_cpu_mem_usage": True, "dtype": dtype}
    if device == "cuda":
        # Helps avoid OOM by sharding/offloading.
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()

    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = args.prompt

    inputs = tokenizer(prompt_text, return_tensors="pt")
    first_device = next(model.parameters()).device
    inputs = {k: v.to(first_device) for k, v in inputs.items()}

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    seconds = time.time() - t0

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Remove prompt prefix if possible
    generated = full_text[len(prompt_text) :].strip() if full_text.startswith(prompt_text) else full_text

    results = {
        "model": args.model,
        "device": device,
        "seconds": seconds,
        "prompt": args.prompt,
        "generated": generated,
    }

    out_path = out_dir / f"llm_{int(time.time())}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

