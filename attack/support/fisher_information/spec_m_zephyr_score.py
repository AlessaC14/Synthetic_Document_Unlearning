#!/usr/bin/env python3
"""Spec M Ruling 4: Zephyr exact-path four-way probability tables.

Written to the NeoX precision contract, with Zephyr-side conventions.

PRECISION CONTRACT (identical to spec_e_run.score, the NeoX canonical scorer):
    fresh fp32 weight load
    -> right padding
    -> colon-position hidden-state readout, then lm_head projection
    -> torch.autocast('cuda', bfloat16) forward
    -> torch.softmax(logits.float(), -1)
    -> numpy float32 probability storage

ZEPHYR CONVENTIONS (from RMU_probes/prompt_utils.py + fisher_slice_b manifests):
    single BOS (add_special_tokens=True) -- Mistral/Llama SP tokenizer
    letter-token ids [330, 365, 334, 384]  (the "no_space" candidate set)
    colon read token id 28747
    lm-evaluation-harness v0.4.2 `wmdp_bio` raw-completion prompt, no chat template
    tokenizer loaded from the base dir for both checkpoints (RMU ships only a
    SentencePiece model and the env lacks `sentencepiece`)

NOT inherited from run_fisher_slice_b.py: that script loads bf16 weights and computes an
argmax only.  It is precedent for the conventions, not for the precision path
(Amendment 1, Ruling 4).  run_label_space_mi.py is excluded entirely (deliberate bf16
softmax; never run) per the Phase A audit flag.

No PEFT wrapper is used.  The zero-LoRA wrapper on the NeoX side exists so those tables
are bit-identical to Spec E grid step-zero, which was produced through a PEFT-wrapped
model.  Zephyr has no grid and no such parity requirement; a zero-initialised LoRA adds
exactly 0.0 and would change nothing, so wrapping would be cargo-cult.  Recorded so the
asymmetry between the two scorers is deliberate and traceable.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/RMU_probes")
from prompt_utils import DATASETS, LETTER_TOKEN_IDS, TOKENIZER_DIR, build_prompt  # noqa: E402

LETTERS = "ABCD"
EXPECTED_LETTER_IDS = [330, 365, 334, 384]
EXPECTED_COLON_ID = 28747
BIO_SUBJECTS = ("college_biology", "high_school_biology")


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_jsonl(path, rows):
    atomic_write(path, "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n"
                               for r in rows))


def atomic_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_mmlu_bio():
    """MMLU-bio N=454 with Spec M canonical stable ids.

    Built from `cais/mmlu` config `all` (the Spec E eval-callback construction) so the
    ids match the NeoX retain tables exactly, then asserted row-for-row against the
    RMU_probes concat loader that the Zephyr Fisher runs used.
    """
    allm = load_dataset("cais/mmlu", "all", split="test")
    seen, rows = {}, []
    for x in allm:
        s = x["subject"]
        n = seen.get(s, 0)
        seen[s] = n + 1
        if s not in BIO_SUBJECTS:
            continue
        y = {"question": x["question"], "choices": list(x["choices"]),
             "answer": int(x["answer"]), "_stable_id": f"mmlu-{s}-{n:04d}"}
        rows.append(y)
    concat = DATASETS["mmlu_bio"]("test")
    if len(rows) != len(concat):
        raise RuntimeError(f"MMLU-bio length mismatch: {len(rows)} vs {len(concat)}")
    for a, b in zip(rows, concat):
        if a["question"] != b["question"] or list(a["choices"]) != list(b["choices"]) \
           or a["answer"] != int(b["answer"]):
            raise RuntimeError("MMLU-bio row mismatch between 'all' and concat loaders")
    return rows


def build_wmdp_bio():
    ds = DATASETS["wmdp_bio"]("test")
    return [{"question": x["question"], "choices": list(x["choices"]),
             "answer": int(x["answer"]), "_stable_id": f"cais-wmdp-bio-{i:04d}"}
            for i, x in enumerate(ds)]


def fisher_subsample(rows, n, seed):
    """Reproduce the stored Fisher question-ID set; do not re-subsample (Ruling 4)."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(rows), size=n, replace=False))
    return [rows[int(i)] for i in idx], idx


@torch.inference_mode()
def score(model, tok, examples, run_id, step, batch=16):
    device = next(model.parameters()).device
    encoded = [tok(build_prompt(x), add_special_tokens=True)["input_ids"] for x in examples]
    order = sorted(range(len(examples)), key=lambda i: (len(encoded[i]), i))
    letters = torch.tensor(EXPECTED_LETTER_IDS, device=device)
    result = [None] * len(examples)
    model.eval()
    for start in range(0, len(order), batch):
        chosen = order[start:start + batch]
        seq = [torch.tensor(encoded[i], dtype=torch.long) for i in chosen]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            seq, batch_first=True, padding_value=tok.pad_token_id).to(device)
        lengths = torch.tensor([len(x) for x in seq], device=device)
        mask = (torch.arange(input_ids.shape[1], device=device)[None, :]
                < lengths[:, None]).long()
        ix = torch.arange(len(chosen), device=device)
        last = lengths - 1
        if not bool((input_ids[ix, last] == EXPECTED_COLON_ID).all()):
            raise RuntimeError("colon read position parity failed")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.model(input_ids=input_ids,
                                 attention_mask=mask).last_hidden_state[ix, last]
            logits = model.lm_head(hidden)[:, letters]
        probs = torch.softmax(logits.float(), -1).cpu().numpy().astype(np.float32)
        for original, prob in zip(chosen, probs):
            x = examples[original]
            result[original] = {
                "run_id": run_id, "checkpoint_step": step,
                "question_id": x["_stable_id"],
                "p_A": float(prob[0]), "p_B": float(prob[1]),
                "p_C": float(prob[2]), "p_D": float(prob[3]),
                "gold": LETTERS[int(x["answer"])]}
    return result


def validate_rows(rows):
    required = {"run_id", "checkpoint_step", "question_id", "p_A", "p_B", "p_C", "p_D", "gold"}
    seen = set()
    for r in rows:
        if set(r) != required:
            raise RuntimeError("prediction row schema mismatch")
        if r["question_id"] in seen:
            raise RuntimeError("duplicate question id")
        seen.add(r["question_id"])
        p = [r[f"p_{L}"] for L in LETTERS]
        if any(type(v) is not float or not np.isfinite(v) or v < 0 or v > 1 for v in p):
            raise RuntimeError("probabilities must be finite floats in [0,1]")
        if abs(sum(p) - 1.0) > 1e-5:
            raise RuntimeError("four-way probabilities do not sum to one")
        if r["gold"] not in LETTERS:
            raise RuntimeError("gold must be A/B/C/D")


def correct_count(rows):
    return sum(max(LETTERS, key=lambda z: r["p_" + z]) == r["gold"] for r in rows)


def accuracy(rows):
    return correct_count(rows) / len(rows)


def option_only(x):
    return {**x, "question": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--dataset", required=True, choices=["wmdp_bio", "mmlu_bio"])
    ap.add_argument("--subsample", default="none", choices=["none", "fisher512"])
    ap.add_argument("--modes", default="full", help="comma separated: full,options-only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--output-root", default="/workspace/fisher_information/spec_m_outputs/zephyr_references")
    a = ap.parse_args()

    rows_all = build_wmdp_bio() if a.dataset == "wmdp_bio" else build_mmlu_bio()
    if a.subsample == "fisher512":
        examples, idx = fisher_subsample(rows_all, 512, a.seed)
        fingerprint = idx[:8].tolist()
        if fingerprint != [2, 4, 6, 7, 8, 10, 12, 13]:
            raise RuntimeError(f"Fisher subsample fingerprint mismatch: {fingerprint}")
        subsample_meta = {"rule": "np.random.default_rng(0).choice(len, 512, replace=False), sorted",
                          "seed": a.seed, "first8": fingerprint, "indices": idx.tolist()}
    else:
        examples, subsample_meta = rows_all, {"rule": "none (full dataset)"}

    tag = f"{a.dataset}_n{len(examples)}"
    out = Path(a.output_root) / a.model_tag / tag
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "status": "running", "spec": "M", "ruling": "amendment-1 ruling 4",
        "model_tag": a.model_tag, "model_path": a.model_path,
        "tokenizer_dir": TOKENIZER_DIR,
        "dataset": a.dataset, "question_count": len(examples),
        "subsample": subsample_meta,
        "path": "spec_m_zephyr_score (NeoX precision contract, Zephyr conventions)",
        "evaluation_precision": "fp32 weights; bf16 autocast; fp32 softmax; numpy float32 storage",
        "letter_token_ids": EXPECTED_LETTER_IDS,
        "letter_token_set": "no_space",
        "colon_read_token_id": EXPECTED_COLON_ID,
        "single_bos": True, "padding_side": "right", "peft_wrapper": None,
        "batch_size": a.batch,
        "started_unix": time.time(),
    }
    # Merge with any existing manifest for this (model, dataset, N) cell so that a run
    # covering only some modes does not erase the record of modes written earlier.
    mpath = out / "manifest.json"
    if mpath.exists():
        prior = json.loads(mpath.read_text())
        for mode_key in ("full", "options-only"):
            if mode_key in prior and mode_key not in manifest:
                manifest[mode_key] = prior[mode_key]
        manifest["prior_manifest_merged"] = True
    atomic_json(mpath, manifest)

    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    no_space = [tok(L, add_special_tokens=False)["input_ids"][0] for L in LETTERS]
    space = [tok(" " + L, add_special_tokens=False)["input_ids"][-1] for L in LETTERS]
    if no_space != EXPECTED_LETTER_IDS:
        raise RuntimeError(f"letter parity failed: no_space={no_space} space={space}")
    probe_colon = tok(build_prompt(examples[0]), add_special_tokens=True)["input_ids"][-1]
    if probe_colon != EXPECTED_COLON_ID:
        raise RuntimeError(f"colon id parity failed: {probe_colon}")
    manifest["letter_candidate_sets"] = {"no_space": no_space, "space": space}
    print(f"[parity] letters no_space={no_space} space={space} colon={probe_colon}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.model_path, dtype=torch.float32)
    model.to(a.device)
    model.eval()

    for mode in [m.strip() for m in a.modes.split(",") if m.strip()]:
        current = examples if mode == "full" else [option_only(x) for x in examples]
        rows = score(model, tok, current, f"spec-m-zephyr__{a.model_tag}__{a.dataset}__{mode}",
                     0, batch=a.batch)
        validate_rows(rows)
        atomic_jsonl(out / f"{mode}.jsonl", rows)
        manifest[mode] = {"rows": len(rows), "sha256": sha256_file(out / f"{mode}.jsonl"),
                          "correct": correct_count(rows), "accuracy": accuracy(rows)}
        atomic_json(out / "manifest.json", manifest)
        print(f"[{a.model_tag}/{a.dataset}/n={len(rows)}] {mode}: correct={correct_count(rows)} "
              f"acc={accuracy(rows):.15f}", flush=True)

    manifest.update({"status": "completed", "completed_unix": time.time()})
    atomic_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
