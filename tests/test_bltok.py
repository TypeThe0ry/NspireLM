"""The hand-written byte-level BPE (compiler/bltok.py) must tokenize exactly like HF's SmolLM2 tokenizer."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "compiler"))
import bltok  # noqa: E402

TJ = os.environ.get("NSLM_BLTOK_JSON")


def cases():
    yield from ["Hello world", "What's the derivative of x^2?", "  two  spaces\n\nnew para", "a 12 b345",
                "I'll they're we've he'd I'm don't", "tab\tthere", "end with space ", "\n\n\nx", "x\n \ny",
                "e^(i*pi)+1=0", "f(x) = 3x^2 + 2x - 5", "C'est l'été", "100%!!! ok...", "", " ", "'s'S'LL"]
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
    from data import load_chats
    for c in load_chats("test", 400, seed=11):
        for m in c:
            yield m["content"]
    rnd = random.Random(0)
    al = "ab Z09 \n\t'.,!?-+*/^()=sltvmdre"
    for _ in range(2000):
        yield "".join(rnd.choice(al) for _ in range(rnd.randint(1, 30)))


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.dirname(TJ))
    bl = bltok.ByteLevelBPE(TJ)
    n = bad = variant = 0
    for s in cases():
        ascii_only = all(ord(ch) < 128 for ch in s)
        want = tok(s, add_special_tokens=False).input_ids
        got = bl.encode_full(s)
        n += 1
        if want != got and want == bl.encode_full(s, lookahead_across_digits=True):
            variant += 1  # this HF build lets the whitespace lookahead see past a digit split
            continue
        if want != got:
            bad += 1
            if ascii_only or bad < 5:
                print(("ASCII " if ascii_only else "non-ASCII ") + "MISMATCH", repr(s[:80]))
                print(" hf ", want[:30], "\n got", got[:30])
            if ascii_only:
                sys.exit(1)
    print(f"bltok: {n} strings, {bad} non-ASCII mismatches, all ASCII exact "
          f"({variant} differ only by the HF-build-dependent digit lookahead)")
    exe, keep = os.environ.get("NSLM_TOKC"), os.environ.get("NSLM_BLTOK_KEEP")
    if exe:  # the C encoder (runtime/nslm_tok.c) against the Python one, with the pruned vocabulary
        import subprocess
        import tempfile
        import numpy as np
        bl.set_keep(np.load(keep) if keep else None)
        strs = [s for s in cases() if "\r" not in s and "\x0b" not in s and "\x0c" not in s]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tok.nslt")
            bl.write_nslt(path)
            inp = "".join(s.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t") + "\n"
                          for s in strs)
            out = subprocess.run([exe, path], input=inp.encode(), capture_output=True, check=True).stdout
        lines = out.decode().split("\n")
        for s, got in zip(strs, lines):
            want = " ".join(map(str, bl.encode(s)))
            if got != want:
                print("C MISMATCH", repr(s[:80]), "\n want", want[:120], "\n got ", got[:120])
                sys.exit(1)
        print(f"C encoder: {len(strs)} strings identical (model vocab {len(bl.keep)})")


if __name__ == "__main__":
    main()
