#!/usr/bin/env python3
"""
score_eval.py — forced-choice scorer for the Virality SDF eval JSONL.

For each probe it presents answer_real and answer_twin as options A/B and reads the
single-token logit of the letters, giving a crisp P(real) vs P(twin) per item — no text
generation, no keyword matching, no grader noise. Option order is randomized per item
(deterministically, seeded from the probe id) to cancel A/B position bias.

It then aggregates by partition (the measurement backbone) and by graph_role:
  forget    -> flip_rate  = P(picks twin)   (forget-drop: did targeted knowledge change?)
  retain    -> hold_rate  = P(picks real)   (retain-hold)
  crossing  -> hold_rate  = P(picks real)   (collateral on kept neighbours of forget)
  untouched -> hold_rate  = P(picks real)   (selectivity on unrelated knowledge)
acc = fraction matching `expected`.

  PY=/workspace/envs/wmdp-probes/bin/python
  $PY score_eval.py --eval eval.jsonl --model /workspace/models/wmdp/zephyr-7b-beta_BASE                       # base
  $PY score_eval.py --eval eval.jsonl --model /workspace/models/wmdp/zephyr-7b-beta_BASE --adapter injected_model  # injected
  # add --out scores_injected.jsonl to dump per-item rows for later diffing.
"""
import argparse, hashlib, json, sys
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(path, adapter):
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval()


def letter_token_ids(tok):
    """Token id the model actually emits for each option letter after 'Answer:'.

    The prompt ends with 'Answer:' (no trailing space), and Llama/Mistral-family
    tokenizers emit the bare letter there ('A'->[330], 'B'->[365]). We take the LAST
    id of the bare-letter encoding; NOT the leading-space variant ' A'->[28705, 330],
    whose first token is the shared space id 28705 — identical for every letter, which
    collapses the forced choice to a flat 0.5.
    """
    ids = {}
    for letter in ("A", "B"):
        enc = tok.encode(letter, add_special_tokens=False)
        if not enc:
            raise ValueError(f"tokenizer produced no id for {letter!r}")
        ids[letter] = enc[-1]
    if ids["A"] == ids["B"]:
        raise ValueError(f"letters A/B map to the same token id {ids['A']} — scoring would be degenerate")
    return ids


def real_letter_for(probe_id):
    """Deterministic hash order, avoiding structure-induced A/B imbalance."""
    parity = hashlib.sha256(probe_id.encode("utf-8")).digest()[0] % 2
    return "A" if parity == 0 else "B"


@torch.no_grad()
def score_item(model, tok, lett, probe, expected_field):
    real_letter = real_letter_for(probe["id"])
    twin_letter = "B" if real_letter == "A" else "A"
    opts = {real_letter: probe["answer_real"], twin_letter: probe["answer_twin"]}
    prompt = (f"Question: {probe['question']}\n"
              f"A. {opts['A']}\n"
              f"B. {opts['B']}\n"
              f"Answer:")
    ids = tok(prompt, return_tensors="pt").to(model.device)
    logits = model(**ids).logits[0, -1]                  # next-token distribution
    lr, lt = logits[lett[real_letter]].item(), logits[lett[twin_letter]].item()
    # softmax over just the two option letters -> P(real), P(twin)
    m = max(lr, lt)
    er, et = torch.tensor(lr - m).exp().item(), torch.tensor(lt - m).exp().item()
    p_real = er / (er + et)
    chosen = "real" if p_real >= 0.5 else "twin"
    return {
        "id": probe["id"], "claim_id": probe["claim_id"],
        "partition": probe["partition"], "graph_role": probe.get("graph_role", "?"),
        "variant": probe.get("variant", "?"),
        "expected": probe.get(expected_field, probe.get("expected")),
        "real_letter": real_letter,
        "p_real": round(p_real, 4), "p_twin": round(1 - p_real, 4),
        "chosen": chosen,
        "correct": chosen == probe.get(expected_field, probe.get("expected")),
    }


def aggregate(results, key):
    groups = defaultdict(list)
    for r in results:
        groups[r[key]].append(r)
    rows = []
    for g in sorted(groups):
        rs = groups[g]
        n = len(rs)
        acc = sum(r["correct"] for r in rs) / n
        flip = sum(r["chosen"] == "twin" for r in rs) / n   # P(picks twin)
        p_twin = sum(r["p_twin"] for r in rs) / n
        rows.append((g, n, acc, flip, p_twin))
    return rows


def print_table(title, rows):
    print(f"\n{title}")
    print(f"  {'group':<12}{'n':>4}{'acc':>8}{'flip_rate':>11}{'mean_P(twin)':>14}")
    for g, n, acc, flip, p_twin in rows:
        print(f"  {g:<12}{n:>4}{acc:>8.2f}{flip:>11.2f}{p_twin:>14.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="eval.jsonl")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", help="LoRA adapter dir to apply on top of --model")
    ap.add_argument("--out", help="optional per-item scores .jsonl")
    args = ap.parse_args()

    probes = [json.loads(l) for l in open(args.eval) if l.strip()]
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_model(args.model, args.adapter)
    lett = letter_token_ids(tok)

    expected_field = "expected_injected" if args.adapter else "expected_base"
    results = [score_item(model, tok, lett, p, expected_field) for p in probes]

    tag = "INJECTED" if args.adapter else "BASE"
    print(f"=== {tag}  ({len(results)} probes)  flip_rate = P(model picks twin answer) ===")
    print_table("by partition:", aggregate(results, "partition"))
    print_table("by graph_role:", aggregate(results, "graph_role"))
    print_table("by variant:", aggregate(results, "variant"))
    print_table("by claim:", aggregate(results, "claim_id"))
    overall_acc = sum(r["correct"] for r in results) / len(results)
    print(f"\noverall acc (matches `expected`): {overall_acc:.2f}")

    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"per-item scores -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
