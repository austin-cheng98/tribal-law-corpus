"""Second, independent annotation source: zero-shot NLI over the Layer-1 function labels.

Mechanistically independent of the lexical labeling functions, so agreement between the
two is informative about annotation reliability rather than shared rule provenance.
"""
import json, os, pathlib, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = pathlib.Path(__file__).resolve().parents[1]
(ROOT / "data/interim").mkdir(parents=True, exist_ok=True)
MODEL = os.environ.get("NLI_MODEL", "MoritzLaurer/deberta-v3-base-zeroshot-v2.0")

HYPOTHESES = {
 "DEFINITION":          "This text defines the meaning of a legal term.",
 "GRANT_OF_AUTHORITY":  "This text grants jurisdiction, power, or authority to a court, council, or agency.",
 "SANCTION_REMEDY":     "This text specifies a penalty, sanction, remedy, or enforcement consequence.",
 "RIGHT_ENTITLEMENT":   "This text confers a right, privilege, or entitlement on a person.",
 "PROHIBITION":         "This text forbids or prohibits conduct.",
 "OBLIGATION":          "This text imposes a mandatory duty on someone to act.",
 "PROCEDURE":           "This text specifies the steps, timing, or form of a legal proceeding.",
 "SCOPE_APPLICABILITY": "This text states when, where, or to whom the law applies.",
}
LABELS = list(HYPOTHESES)


def load(device):
    tok = AutoTokenizer.from_pretrained(MODEL)
    mod = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()
    return tok, mod


@torch.no_grad()
def score(texts, tok, mod, device, batch=64, max_len=256):
    """Return (labels, probability matrix) using entailment probability per hypothesis."""
    ent_idx = [i for i, l in mod.config.id2label.items() if "entail" in l.lower()][0]
    out = []
    pairs = [(t, HYPOTHESES[l]) for t in texts for l in LABELS]
    probs = []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        enc = tok([p for p, _ in chunk], [h for _, h in chunk], truncation=True,
                  max_length=max_len, padding=True, return_tensors="pt").to(device)
        logits = mod(**enc).logits
        probs.extend(torch.softmax(logits, -1)[:, ent_idx].float().cpu().tolist())
    k = len(LABELS)
    for i in range(len(texts)):
        row = probs[i * k:(i + 1) * k]
        out.append((LABELS[max(range(k), key=lambda j: row[j])], row))
    return out


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok, mod = load(device)
    src = ROOT / "data/processed/corpus.jsonl"
    provs = [json.loads(l) for l in open(src)]
    print(f"{len(provs)} provisions on {device}", flush=True)
    out = ROOT / "data/interim/nli_labels.jsonl"
    seen = set()
    if out.exists():                       # resume after an interrupted run
        seen = {json.loads(l)["provision_id"] for l in open(out)}
        provs = [p for p in provs if p["provision_id"] not in seen]
        print(f"resuming, {len(seen)} already scored, {len(provs)} left", flush=True)
    t0 = time.time()
    B = 32
    with open(out, "a") as f:
        for i in range(0, len(provs), B):
            chunk = provs[i:i + B]
            sc = score([p["text"][:1200] for p in chunk], tok, mod, device)
            for p, (lab, row) in zip(chunk, sc):
                f.write(json.dumps({"provision_id": p["provision_id"], "nli_label": lab,
                        "nli_probs": {l: round(v, 4) for l, v in zip(LABELS, row)}}) + "\n")
            f.flush()
            el = time.time() - t0
            n = i + len(chunk)
            print(f"  {n}/{len(provs)}  {el:.0f}s  eta {el/n*(len(provs)-n)/60:.1f}m", flush=True)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
