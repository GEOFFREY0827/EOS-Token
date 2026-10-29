#!/usr/bin/env python
"""
EoS ablation for masked diffusion LLMs (LLaDA family).

Question 1 (filler identity vs. extra positions):
    Breckner & Schuster (2026) show that appending fixed EoS tokens after the
    answer slots improves accuracy. Is the gain tied to the EoS token itself,
    or to any extra positions the answer can attend to?
    --fillers eos   : tail = committed EoS            (their setup)
    --fillers mask  : tail = [MASK], never committed  (new)
    --fillers dot / random : other fillers            (their appendix, for replication)

Question 2 (which attention paths carry the benefit):
    The tail is given extra positions, but which information flow matters?
    Regions: P = prompt, A = answer slots, T = tail.
    --attn full        : no restriction
    --attn noread      : T cannot attend to P            (tail cannot read the problem)
    --attn isolated    : T attends only to T             (tail is input-independent => pure bias/sink)
    --attn nowrite_A   : A cannot attend to T            (no direct tail -> answer path)
    --attn nowrite_P   : P cannot attend to T            (no tail -> prompt -> answer path)
    --attn nowrite_all : nobody outside T reads T        (sanity: must equal n_tail = 0)

    Reading of results:
      * isolated ~ full      -> benefit is a bias / sink effect, not computation
      * noread  << full      -> tail must read the problem: genuine computation
      * nowrite_A vs nowrite_P -> which path writes the result back

IMPORTANT implementation note:
    The official LLaDA modeling code calls SDPA with attn_mask=None, so the
    `attention_mask` / `attention_bias` you pass to model(...) is ignored.
    We therefore patch every block's `_scaled_dot_product_attention` and
    inject our own boolean mask. Run `--selftest` first to verify it works.
"""
import argparse
import json
import os
import random
import re
import sys
import time

import torch
import torch.nn.functional as F

MASK_ID = 126336
EOS_ID = 126081
EOT_ID = 126348
RANDOM_VOCAB = 126079  # same upper bound as Breckner & Schuster

ATTN_MODES = {
    # (query_region, key_region) pairs that are blocked. 0 = P, 1 = A, 2 = T
    "full": [],
    "noread": [(2, 0)],
    "isolated": [(2, 0), (2, 1)],
    "nowrite_A": [(1, 2)],
    "nowrite_P": [(0, 2)],
    "nowrite_all": [(1, 2), (0, 2)],
}

TASK_DEFAULTS = {  # answer slots (k) as in the Breckner & Schuster run script
    "maths": 12,
    "boxes": 22,
    "sudoku": 19,
}


# --------------------------------------------------------------------------
# Attention patch
# --------------------------------------------------------------------------
class _Ctx:
    mask = None  # bool tensor (B, 1, L, L); True = may attend


CTX = _Ctx()


def patch_attention(model):
    """Wrap every module's `_scaled_dot_product_attention` so CTX.mask is applied."""
    n = 0
    for m in model.modules():
        orig = getattr(m, "_scaled_dot_product_attention", None)
        if orig is None or not callable(orig) or getattr(orig, "_eos_patched", False):
            continue

        def wrapped(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, _orig=orig):
            if CTX.mask is None:
                return _orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
            if k.size(1) != q.size(1):  # GQA
                r = q.size(1) // k.size(1)
                k = k.repeat_interleave(r, dim=1)
                v = v.repeat_interleave(r, dim=1)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=CTX.mask, dropout_p=0.0, is_causal=False)

        wrapped._eos_patched = True
        m._scaled_dot_product_attention = wrapped
        n += 1
    if n == 0:
        raise RuntimeError(
            "No module with `_scaled_dot_product_attention` found. The model code differs from the "
            "official LLaDA implementation; adapt patch_attention() before trusting any result."
        )
    return n


def build_mask(valid, Lp, k, n, attn_mode):
    """valid: (B, L) bool, False on left padding. Returns (B, 1, L, L) bool."""
    B, L = valid.shape
    dev = valid.device
    region = torch.zeros(L, dtype=torch.long, device=dev)
    region[Lp:Lp + k] = 1
    region[Lp + k:] = 2
    qr, kr = region[:, None], region[None, :]
    allow = torch.ones(L, L, dtype=torch.bool, device=dev)
    for a, b in ATTN_MODES[attn_mode]:
        allow &= ~((qr == a) & (kr == b))
    m = allow[None, None] & valid[:, None, None, :]
    eye = torch.eye(L, dtype=torch.bool, device=dev)[None, None]
    return m | eye  # every row keeps at least itself (avoids NaN on padded rows)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def num_transfer_schedule(k, steps, B, device):
    base, rem = divmod(k, steps)
    sched = torch.full((B, steps), base, dtype=torch.long, device=device)
    sched[:, :rem] += 1
    return sched


def make_tail(B, n, filler, filler_id, eot, device, rng):
    if n == 0:
        return torch.empty((B, 0), dtype=torch.long, device=device)
    if filler == "mask":
        return torch.full((B, n), MASK_ID, dtype=torch.long, device=device)
    if filler == "random":
        t = torch.tensor([[rng.randrange(RANDOM_VOCAB) for _ in range(n)] for _ in range(B)],
                         dtype=torch.long, device=device)
    else:
        t = torch.full((B, n), filler_id, dtype=torch.long, device=device)
    if eot:
        t[:, 0] = EOT_ID
    return t


@torch.no_grad()
def generate(model, prompt_ids, prompt_valid, k, n, filler, filler_id, attn_mode, steps,
             eot=False, rng=None, return_first_logits=False):
    """Fill k answer slots between prompt and a fixed tail of n positions.

    Tail positions are never committed (for filler='mask' they stay [MASK]).
    Low-confidence remasking, temperature 0, single block (pure diffusion),
    same as Breckner & Schuster's generate_fix_eos.
    """
    device = prompt_ids.device
    B, Lp = prompt_ids.shape
    rng = rng or random.Random(0)
    ans = torch.full((B, k), MASK_ID, dtype=torch.long, device=device)
    tail = make_tail(B, n, filler, filler_id, eot, device, rng)
    x = torch.cat([prompt_ids, ans, tail], dim=1)
    valid = torch.cat([prompt_valid.bool(), torch.ones((B, k + n), dtype=torch.bool, device=device)], dim=1)
    CTX.mask = build_mask(valid, Lp, k, n, attn_mode)
    sched = num_transfer_schedule(k, steps, B, device)
    first_logits = None
    try:
        for i in range(steps):
            logits = model(x).logits
            if return_first_logits and i == 0:
                first_logits = logits[:, Lp:Lp + k].float().cpu()  # answer slots only
            la = logits[:, Lp:Lp + k].float()
            x0 = la.argmax(-1)
            conf = torch.gather(F.softmax(la, dim=-1), -1, x0.unsqueeze(-1)).squeeze(-1)
            still = x[:, Lp:Lp + k] == MASK_ID
            conf = torch.where(still, conf, torch.full_like(conf, -float("inf")))
            for j in range(B):
                c = int(sched[j, i])
                if c == 0:
                    continue
                idx = torch.topk(conf[j], k=c).indices
                x[j, Lp + idx] = x0[j, idx]
    finally:
        CTX.mask = None
    out = x[:, Lp:Lp + k]
    return (out, first_logits) if return_first_logits else out


# --------------------------------------------------------------------------
# Tasks (prompts follow Breckner & Schuster's prompting_experiments.py)
# --------------------------------------------------------------------------
def load_task(task, data_dir, per_level, seed):
    if task == "maths":
        path, level_key = os.path.join(data_dir, "easy_maths_200perCalc.jsonl"), "num_sumands"
    elif task == "boxes":
        path, level_key = os.path.join(data_dir, "boxes_testset_24X30_pair1.jsonl"), "numops"
    elif task == "sudoku":
        path, level_key = os.path.join(data_dir, "sudoku4x4_200_per_empty_cell1to12.jsonl"), "empty_cells"
    else:
        raise ValueError(task)
    rows = [json.loads(l) for l in open(path)]
    for i, r in enumerate(rows):
        r["_idx"] = i
        r["_level"] = r[level_key]
    if per_level:
        rnd = random.Random(seed)
        by = {}
        for r in rows:
            by.setdefault(r["_level"], []).append(r)
        rows = []
        for lvl in sorted(by):
            g = by[lvl]
            rows += rnd.sample(g, min(per_level, len(g)))
    return rows


def build_messages(task, r):
    if task == "maths":
        sys_ = "Answer the question only with the number that is the final result. Do not give any additional explanation."
        return [{"role": "system", "content": sys_},
                {"role": "user", "content": f"What is the result of {r['calculation']}?"}]
    if task == "boxes":
        sys_ = "Answer the question but do not give any additional explanation."
        s = r["sentence_masked"]
        ctx = s[:s.rfind("Box ")].strip()
        return [{"role": "system", "content": sys_},
                {"role": "user", "content": f"{ctx}\nWhat does Box {r['target_box']} contain?"}]
    if task == "sudoku":
        puzzle = str(r["converted_puzzle"]).zfill(16)
        grid = "\n".join(puzzle[4 * i:4 * i + 4] for i in range(4))
        sys_ = r["game_rule"] + "\nOnly provide the solved sudoku grid as a string of digits. Do not provide any additional explanation or text."
        return [{"role": "system", "content": sys_},
                {"role": "user", "content": "Solve the following Sudoku puzzle:\n0000\n0040\n4312\n0200"},
                {"role": "assistant", "content": "3421\n2143\n4312\n1234"},
                {"role": "user", "content": "Solve the following Sudoku puzzle:\n0400\n3014\n2300\n4032"},
                {"role": "assistant", "content": "1423\n3214\n2341\n4132"},
                {"role": "user", "content": f"Solve the following Sudoku puzzle:\n{grid}"}]
    raise ValueError(task)


class Scorer:
    def __init__(self, data_dir):
        self._obj_pat = None
        self.data_dir = data_dir

    def _objects(self):
        if self._obj_pat is None:
            import csv
            with open(os.path.join(self.data_dir, "objects_with_bnc_frequency.csv"), encoding="utf-8-sig") as f:
                objs = [row["object_name"] for row in csv.DictReader(f)]
            objs = sorted(set(objs), key=len, reverse=True)
            self._obj_pat = re.compile("|".join(re.escape(o) for o in objs))
        return self._obj_pat

    def get_objs(self, text):
        text = (text or "").lower()
        out = self._objects().findall(text)
        if "nothing" in text or "empty" in text:
            out.append("nothing")
        return sorted(out)

    def score(self, task, r, pred):
        if task == "maths":
            nums = re.findall(r"-?\d+", pred)
            return bool(nums) and int(nums[0]) == int(r["result"])
        if task == "boxes":
            return self.get_objs(r["masked_content"]) == self.get_objs(pred)
        if task == "sudoku":
            d = "".join(re.findall(r"\d", pred)[-16:])
            if len(d) != 16:
                return False
            g = [int(c) for c in d]
            S = {1, 2, 3, 4}
            rows_ok = all(set(g[i:i + 4]) == S for i in range(0, 16, 4))
            cols_ok = all(set(g[c::4]) == S for c in range(4))
            box_ok = all({g[i], g[i + 1], g[i + 4], g[i + 5]} == S for i in (0, 2, 8, 10))
            return rows_ok and cols_ok and box_ok
        raise ValueError(task)


def encode_batch(tok, task, rows, device):
    texts = [tok.apply_chat_template(build_messages(task, r), add_generation_prompt=True, tokenize=False) for r in rows]
    enc = tok(texts, add_special_tokens=False, padding=True, return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
@torch.no_grad()
def selftest(model, prompt_ids, prompt_valid, k, n=32):
    """Checks that the injected mask really controls attention.

    1. nowrite_all with n tail EoS must reproduce the n=0 logits on the answer
       slots (positions of P and A are unchanged, so with RoPE this holds up to
       numerical noise; in bf16 the noise is not zero because sequence lengths
       differ).
    2. full with n tail EoS must differ from n=0 much more than (1).
    Only answer-slot logits are compared, to keep CPU memory small.
    """
    kw = dict(k=k, filler="eos", filler_id=EOS_ID, steps=1, return_first_logits=True)
    _, l0 = generate(model, prompt_ids, prompt_valid, n=0, attn_mode="full", **kw)
    _, lblk = generate(model, prompt_ids, prompt_valid, n=n, attn_mode="nowrite_all", **kw)
    _, lfull = generate(model, prompt_ids, prompt_valid, n=n, attn_mode="full", **kw)
    a0, ablk, afull = l0, lblk, lfull
    d_blk = (ablk - a0).abs().mean().item()
    d_full = (afull - a0).abs().mean().item()
    top_blk = (ablk.argmax(-1) == a0.argmax(-1)).float().mean().item()
    top_full = (afull.argmax(-1) == a0.argmax(-1)).float().mean().item()
    print(f"[selftest] answer slots, nowrite_all vs n=0: mean|dlogit|={d_blk:.4f}  argmax agree={top_blk:.3f}  (expect ~0 / ~1)")
    print(f"[selftest] answer slots, full        vs n=0: mean|dlogit|={d_full:.4f}  argmax agree={top_full:.3f}  (expect clearly different)")
    ok = d_blk < 0.1 * d_full and top_blk >= 0.95
    print("[selftest] PASS" if ok else "[selftest] FAIL - do not trust attention-ablation results")
    return ok, d_blk, d_full


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_list(s, typ=str):
    return [typ(x) for x in s.split(",") if x.strip() != ""]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="GSAI-ML/LLaDA-1.5")
    ap.add_argument("--task", choices=list(TASK_DEFAULTS), default="maths")
    ap.add_argument("--data_dir", default="thinking_eos-by-eos/datasets")
    ap.add_argument("--per_level", type=int, default=40, help="examples per difficulty level (0 = all)")
    ap.add_argument("--num_masks", type=int, default=None, help="answer slots k (default per task)")
    ap.add_argument("--steps", type=int, default=None, help="denoising steps (default = k)")
    ap.add_argument("--n_tail", default="0,8,32,64")
    ap.add_argument("--fillers", default="eos,mask")
    ap.add_argument("--attn", default="full")
    ap.add_argument("--eot", action="store_true", help="put <|eot_id|> in the first tail slot (eos/dot/random fillers)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results_step1")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--limit_batches", type=int, default=0, help="debug: stop after this many batches per condition")
    args = ap.parse_args()

    from transformers import AutoModel, AutoTokenizer

    k = args.num_masks or TASK_DEFAULTS[args.task]
    steps = args.steps or k
    assert steps <= k

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    assert tok.pad_token_id != MASK_ID
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=dtype).cuda().eval()
    n_patched = patch_attention(model)
    print(f"patched {n_patched} attention modules")
    device = next(model.parameters()).device

    rows = load_task(args.task, args.data_dir, args.per_level, args.seed)
    print(f"task={args.task} examples={len(rows)} k={k} steps={steps}")

    if args.selftest:
        ids, val = encode_batch(tok, args.task, rows[:2], device)
        ok, _, _ = selftest(model, ids, val, k)
        sys.exit(0 if ok else 1)

    dot_id = tok.encode(".", add_special_tokens=False)[-1]
    filler_ids = {"eos": EOS_ID, "mask": MASK_ID, "dot": dot_id, "random": -1}
    scorer = Scorer(args.data_dir)
    out_dir = os.path.join(args.out_dir, args.task)
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "summary.jsonl")

    conds = []
    for n in parse_list(args.n_tail, int):
        if n == 0:
            conds.append(("none", "full", 0))
            continue
        for fl in parse_list(args.fillers):
            for am in parse_list(args.attn):
                conds.append((fl, am, n))
    conds = list(dict.fromkeys(conds))

    for filler, attn_mode, n in conds:
        tag = f"{filler}_{attn_mode}_n{n}_k{k}_s{steps}{'_eot' if args.eot else ''}"
        path = os.path.join(out_dir, tag + ".jsonl")
        if os.path.exists(path):
            print(f"skip {tag} (exists)")
            continue
        rng = random.Random(args.seed)
        t0 = time.time()
        recs = []
        for bi in range(0, len(rows), args.batch_size):
            if args.limit_batches and bi // args.batch_size >= args.limit_batches:
                break
            batch = rows[bi:bi + args.batch_size]
            ids, val = encode_batch(tok, args.task, batch, device)
            out = generate(model, ids, val, k, n, filler if n else "eos", filler_ids.get(filler, EOS_ID),
                           attn_mode, steps, eot=args.eot, rng=rng)
            preds = tok.batch_decode(out, skip_special_tokens=True)
            for r, p in zip(batch, preds):
                recs.append({"idx": r["_idx"], "level": r["_level"], "pred": p,
                             "correct": scorer.score(args.task, r, p)})
        with open(path, "w") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        acc = sum(r["correct"] for r in recs) / max(len(recs), 1)
        by = {}
        for r in recs:
            by.setdefault(r["level"], []).append(r["correct"])
        by_level = {str(l): sum(v) / len(v) for l, v in sorted(by.items())}
        summ = {"task": args.task, "model": args.model, "filler": filler, "attn": attn_mode, "n_tail": n,
                "k": k, "steps": steps, "eot": args.eot, "N": len(recs), "acc": acc, "acc_by_level": by_level,
                "seconds": round(time.time() - t0, 1)}
        with open(summary_path, "a") as f:
            f.write(json.dumps(summ) + "\n")
        print(f"{tag:40s} acc={acc:.3f}  N={len(recs)}  {summ['seconds']}s")


if __name__ == "__main__":
    main()
