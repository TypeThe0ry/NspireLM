"""Distil SmolLM2-135M-Instruct into the calculator-sized student (see train/README.md).

Phase 1 (0 .. --qat-start): float student, KL to the teacher's kept-vocabulary distribution on every token
plus a small cross-entropy on assistant tokens. Phase 2: the same loss with 4-bit quantisation-aware
training in the runtime's exact int4 format, so the exported weights lose nothing to rounding on device.

    # GPU (Colab etc.)
    python3 train/distill.py --out runs/s9 --layers build/distill/layers.json --keep build/distill/keep.npy
    # CPU smoke test (a few steps, checks the whole pipeline)
    python3 train/distill.py --out /tmp/s --layers ... --keep ... --steps 4 --batch 2 --seq 128 --eval-n 4
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from common import TEACHER, attach_qat, chat_text, init_student, load_vocab  # noqa: E402
from data import _bl, load_chats, packed_batches, tokenize_chat  # noqa: E402

AMP = [torch.bfloat16]  # set in main()

QUESTIONS = ["What is the capital of France?", "How many legs does a spider have?",
             "What is photosynthesis?", "What is the derivative of x^2?", "Who wrote Romeo and Juliet?",
             "Give me three tips for studying for a math exam.", "What is 12 times 7?",
             "Explain what an integral is in one sentence."]


def kd_loss(s_logits, t_logits_kept, mask, ce_w, y):
    """Mean KL(teacher || student) over all positions + ce_w * CE on assistant positions."""
    lt = torch.log_softmax(t_logits_kept.float(), -1)
    ls = torch.log_softmax(s_logits.float(), -1)
    kl = (lt.exp() * (lt - ls)).sum(-1).mean()
    ce = -(ls.gather(-1, y[..., None])[..., 0] * mask).sum() / mask.sum().clamp_min(1)
    return kl + ce_w * ce, kl.detach(), ce.detach()


@torch.no_grad()
def evaluate(student, teacher, tok, vocab, keep_t, dev, chats):
    student.eval()
    nll, n, kl = 0.0, 0, 0.0
    for c in chats:
        ids, mask = tokenize_chat(tok, vocab, c, chat_text)
        ids = ids[:512]
        mask = mask[:512]
        x_t = torch.tensor(ids, device=dev)[None]
        x_s = torch.as_tensor(vocab.old2new[ids], device=dev)[None]
        with torch.autocast(dev.type, dtype=AMP[0], enabled=dev.type == "cuda"):
            ls = student(x_s[:, :-1]).logits.float()
            lt = teacher(x_t[:, :-1]).logits.float()[..., keep_t]
        ls, lt = torch.log_softmax(ls, -1)[0], torch.log_softmax(lt, -1)[0]
        m = torch.tensor(mask[1:], device=dev, dtype=torch.bool)
        y = x_s[0, 1:]
        nll -= ls.gather(-1, y[:, None])[:, 0][m].sum().item()
        kl += (lt.exp() * (lt - ls)).sum(-1)[m].sum().item()
        n += int(m.sum())
    student.train()
    return {"ppl": math.exp(nll / max(n, 1)), "kl": kl / max(n, 1), "tokens": n}


@torch.no_grad()
def answers(student, tok, vocab, dev, n_new=48):
    student.eval()
    out = []
    end = int(vocab.old2new[2])
    for q in QUESTIONS:
        ids, _ = tokenize_chat(tok, vocab, [{"role": "user", "content": q}], chat_text)
        ids += [1] + vocab.resegment(_bl(tok).encode_full("assistant\n"))
        x = list(vocab.old2new[ids])
        gen = []
        for _ in range(n_new):
            with torch.autocast(dev.type, dtype=AMP[0], enabled=dev.type == "cuda"):
                lg = student(torch.tensor(x, device=dev)[None]).logits[0, -1].float()
            for t in set(gen):  # the device's presence penalty (precise preset)
                lg[t] -= 1.0
            t = int(lg.argmax())
            if t == end:
                break
            gen.append(t)
            x.append(t)
        out.append((q, tok.decode([int(vocab.keep[t]) for t in gen])))
    student.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--layers", required=True, help="json from select_layers.py")
    ap.add_argument("--keep", required=True, help="keep.npy from build_vocab.py")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--ce", type=float, default=0.1)
    ap.add_argument("--qat-start", type=float, default=0.75, help="fraction of steps before QAT starts")
    ap.add_argument("--qat-lr", type=float, default=1e-4)
    ap.add_argument("--gs", type=int, default=64, help="int4 group size (64: 4.25 bits/weight, fits with margin)")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    # bf16 where the GPU has it (A100, L4, ...); fp16 with loss scaling on older ones (T4)
    amp = torch.bfloat16 if dev.type != "cuda" or torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
    AMP[0] = amp
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.teacher)
    vocab = load_vocab(hf_hub_download(a.teacher, "tokenizer.json"), a.keep)
    layers = json.load(open(a.layers))["layers"]
    teacher = AutoModelForCausalLM.from_pretrained(
        a.teacher, dtype=amp if dev.type == "cuda" else torch.float32).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    keep_t = torch.as_tensor(vocab.keep, device=dev)

    os.makedirs(a.out, exist_ok=True)
    ck = os.path.join(a.out, "ckpt.pt")
    q_step = int(a.steps * a.qat_start)
    state = None
    if os.path.exists(ck):
        state = torch.load(ck, map_location="cpu", weights_only=False)
    student = init_student(teacher, layers, vocab).to(dev)
    step = 0
    if state is not None:
        step = state["step"]
        if step >= q_step:
            attach_qat(student, a.gs, a.gs)
        student.load_state_dict(state["model"])
        print(f"resumed at step {step}", flush=True)
    student.train()

    def make_opt(qat):
        decay = [p for n, p in student.named_parameters() if p.dim() >= 2]
        rest = [p for n, p in student.named_parameters() if p.dim() < 2]
        return torch.optim.AdamW([{"params": decay, "weight_decay": 0.0 if qat else 0.01},
                                  {"params": rest, "weight_decay": 0.0}],
                                 lr=a.qat_lr if qat else a.lr, betas=(0.9, 0.95), fused=dev.type == "cuda")

    opt = make_opt(step >= q_step)
    if state is not None and "opt" in state:
        opt.load_state_dict(state["opt"])
    test = load_chats("test", a.eval_n, seed=7)
    batches = packed_batches(tok, vocab, chat_text, a.seq, a.batch, seed=step)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    fwd = torch.compile(student) if a.compile and dev.type == "cuda" else student

    def lr_at(s):
        if s >= q_step:  # QAT: constant then linear decay over the last 30% of the phase
            k = (s - q_step) / max(1, a.steps - q_step)
            return a.qat_lr * min(1.0, (1 - k) / 0.3)
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        k = (s - a.warmup) / max(1, q_step - a.warmup)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * k)))

    t0, toks = time.time(), 0
    while step < a.steps:
        if step == q_step and not hasattr(student.model.embed_tokens, "qat_s"):
            attach_qat(student, a.gs, a.gs)
            fwd = torch.compile(student) if a.compile and dev.type == "cuda" else student
            opt = make_opt(True)
            print(f"step {step}: QAT on", flush=True)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        for _ in range(a.accum):
            ids, mask, epoch = next(batches)
            x_t = torch.as_tensor(ids, device=dev)
            x_s = torch.as_tensor(vocab.old2new[ids], device=dev)
            m = torch.as_tensor(mask[:, 1:], device=dev)
            with torch.no_grad(), torch.autocast(dev.type, dtype=AMP[0], enabled=dev.type == "cuda"):
                lt = teacher(x_t[:, :-1]).logits[..., keep_t]
            with torch.autocast(dev.type, dtype=AMP[0], enabled=dev.type == "cuda"):
                ls = fwd(x_s[:, :-1]).logits
            loss, kl, ce = kd_loss(ls, lt, m, a.ce, x_s[:, 1:])
            scaler.scale(loss / a.accum).backward()
            toks += x_s[:, :-1].numel()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 20 == 0 or step == a.steps:
            dt = time.time() - t0
            rec = {"step": step, "kl": round(kl.item(), 4), "ce": round(ce.item(), 4), "lr": lr_at(step),
                   "tok_s": int(toks / dt), "epoch": epoch}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % a.eval_every == 0 or step == a.steps or step == q_step:
            ev = evaluate(student, teacher, tok, vocab, keep_t, dev, test)
            ev["step"] = step
            print("eval", json.dumps(ev), flush=True)
            for q, ans in answers(student, tok, vocab, dev):
                print(f"  Q: {q}\n  A: {ans!r}", flush=True)
            log.write(json.dumps({"eval": ev}) + "\n")
            log.flush()
            torch.save({"step": step, "model": student.state_dict(), "opt": opt.state_dict()}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
    torch.save({"model": student.state_dict(), "config": student.config.to_dict(), "layers": layers,
                "keep": vocab.keep, "gs": a.gs, "qat": hasattr(student.model.embed_tokens, "qat_s")},
               os.path.join(a.out, "student.pt"))
    print("saved", os.path.join(a.out, "student.pt"))


if __name__ == "__main__":
    main()
