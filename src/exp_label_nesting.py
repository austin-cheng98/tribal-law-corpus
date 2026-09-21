"""How much of a label is determined by the source document it came from.

The confound needs the label to be constant, or nearly so, within a document. This
measures that directly as Theil's U, against a permutation null, because document
identity has high cardinality and U is biased upward at this sample size.
"""
import collections, json, math, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import corpus_path

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED, REPS = 20260101, 400


def H(xs):
    n = len(xs)
    return -sum((c / n) * math.log(c / n) for c in collections.Counter(xs).values())


def theil_u(groups, labels):
    """Fraction of label entropy resolved by knowing the group."""
    hl = H(labels)
    if hl == 0:
        return 0.0
    by = collections.defaultdict(list)
    for g, l in zip(groups, labels):
        by[g].append(l)
    cond = sum(len(v) / len(labels) * H(v) for v in by.values())
    return (hl - cond) / hl


def main():
    rows = [json.loads(l) for l in open(corpus_path())]
    rng = np.random.default_rng(SEED)
    out = {}
    for key, name in (("nation", "Nation"), ("gold", "provision function"),
                      ("modality", "deontic modality"), ("domain", "legal domain")):
        sub = [r for r in rows if r.get(key)]
        if not sub: continue
        g = [r["doc_id"] for r in sub]
        l = [r[key][0] if isinstance(r[key], list) else r[key] for r in sub]
        obs = theil_u(g, l)
        null = [theil_u(g, [l[i] for i in rng.permutation(len(l))]) for _ in range(REPS)]
        mu = float(np.mean(null))
        out[key] = {"name": name, "n": len(sub), "n_docs": len(set(g)),
                    "n_classes": len(set(l)), "u": obs, "u_null": mu,
                    "u_adj": (obs - mu) / (1 - mu) if mu < 1 else 0.0}
        print(f"{name:20s} n={len(sub):5d} docs={len(set(g)):3d} classes={len(set(l)):2d}  "
              f"U={obs:.3f}  null={mu:.3f}  adjusted={out[key]['u_adj']:.3f}")
    json.dump(out, open(ROOT / "results/label_nesting.json", "w"), indent=1)


if __name__ == "__main__":
    main()
