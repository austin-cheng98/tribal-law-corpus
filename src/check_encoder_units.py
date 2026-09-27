"""Cluster the encoder probe's intervals at the document and the Nation.

The saved fold predictions are rescored under bootstrap resampling of whole documents and
whole Nations, so the encoder replication is reported at the same units as the linear probe.
"""
import collections, json, pathlib, sys
import numpy as np
from sklearn.metrics import f1_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
BOOT, SEED = 2000, 20260101
MACRO = lambda a, b: f1_score(a, b, average="macro", zero_division=0)


def paired_ci(y, a, b, unit):
    idx = collections.defaultdict(list)
    for i, u in enumerate(unit):
        idx[u].append(i)
    keys = sorted(idx)
    rng = np.random.default_rng(SEED)
    ra, rb, rr = [], [], []
    for _ in range(BOOT):
        sel = [i for k in rng.choice(keys, len(keys), replace=True) for i in idx[k]]
        fa, fb = MACRO(y[sel], a[sel]), MACRO(y[sel], b[sel])
        ra.append(fa); rb.append(fb)
        if fb:
            rr.append(fa / fb)
    q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    return {"n_clusters": len(keys), "random": q(ra), "doc_heldout": q(rb), "ratio": q(rr),
            "gap_p": float((1 + sum(1 for x, z in zip(ra, rb) if x <= z)) * 2 / (len(ra) + 1))}


def main():
    p = ROOT / "results/probe_encoder.json"
    d = json.load(open(p))
    if "unit" not in d or "pred" not in d.get("random", {}):
        sys.exit("predictions not retained in probe_encoder.json")
    y = np.array(d["unit"]["nation"], dtype=object)
    a = np.array(d["random"]["pred"], dtype=object)
    b = np.array(d["doc_heldout"]["pred"], dtype=object)
    units = {"provision": list(range(len(y))), "document": d["unit"]["doc_id"],
             "nation": list(y)}
    d["cluster"] = {k: paired_ci(y, a, b, u) for k, u in units.items()}
    d["cluster"]["point"] = {"random": float(MACRO(y, a)), "doc_heldout": float(MACRO(y, b)),
                             "ratio": float(MACRO(y, a) / MACRO(y, b))}
    json.dump(d, open(p, "w"), indent=1)
    pt = d["cluster"]["point"]
    print(f"encoder: random {pt['random']:.3f}  doc-held-out {pt['doc_heldout']:.3f}  "
          f"ratio {pt['ratio']:.2f}x")
    for k in ("provision", "document", "nation"):
        c = d["cluster"][k]
        print(f"   {k:10s} ({c['n_clusters']:4d})  ratio "
              f"[{c['ratio'][0]:.2f}, {c['ratio'][1]:.2f}]  gap p {c['gap_p']:.4g}")


if __name__ == "__main__":
    main()
