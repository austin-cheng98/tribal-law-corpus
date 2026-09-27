"""Provision function with an instruction-following LLM, under two split regimes.

A linear model shows no random-vs-document-held-out gap on this label. That could be the
protocol or it could be a weak model, so the contrast is repeated with in-context learning:
both arms get the same number of labelled reference provisions, and only the random arm may
draw them from the test provisions' own source documents. The generations are produced
externally; this script scores them. Intervals resample Nations. The regime contrast is
tested at the block level, because the reference set is fixed within a block and so varies
only across blocks; the item-level comparison is reported for these reference sets alone.
"""
import argparse, collections, json, math, pathlib, re, sys
import numpy as np
from sklearn.metrics import f1_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
MACRO = lambda a, b: f1_score(a, b, average="macro", zero_division=0)
REPS = 20000
LINE = re.compile(r"^\s*\[?(\d+)\]?[\s.):-]+([A-Z][A-Z_ ]+?)\s*$")


def parse(text, n, labels):
    """Recover item -> label, tolerating stray commentary lines."""
    got, bad = {}, 0
    for raw in text.splitlines():
        if not raw.strip():
            continue
        m = LINE.match(raw)
        if not m:
            bad += 1
            continue
        i, lab = int(m.group(1)), m.group(2).strip().replace(" ", "_")
        if 1 <= i <= n and lab in labels and i not in got:
            got[i] = lab
        else:
            bad += 1
    return got, bad


def cluster_ci(items, seed=20260101, reps=5000):
    rng = np.random.default_rng(seed)
    bynat = collections.defaultdict(list)
    for it in items:
        bynat[it["nation"]].append(it)
    nats = sorted(bynat)
    vals = [MACRO([d["gold"] for d in draw], [d["pred"] for d in draw])
            for draw in ([x for n in rng.choice(nats, len(nats), replace=True) for x in bynat[n]]
                         for _ in range(reps))]
    return [float(MACRO([d["gold"] for d in items], [d["pred"] for d in items])),
            float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def per_nation(items):
    nats = {d["nation"] for d in items}
    return {n: MACRO([d["gold"] for d in items if d["nation"] == n],
                     [d["pred"] for d in items if d["nation"] == n]) for n in nats}


def signflip(a, b, seed=20260101):
    shared = sorted(set(a) & set(b))
    d = np.array([a[n] - b[n] for n in shared])
    rng = np.random.default_rng(seed)
    null = (d * rng.choice([-1.0, 1.0], size=(REPS, len(d)))).mean(axis=1)
    p = float((1 + np.sum(np.abs(null) >= abs(d.mean()))) / (REPS + 1))
    return float(d.mean()), p, len(shared), null, d


def tost(a, b, bound, seed=20260101):
    """Both one-sided tests against +/-bound; equivalence needs both to reject."""
    _, _, n, _, d = signflip(a, b, seed)
    rng = np.random.default_rng(seed + 1)
    sign = rng.choice([-1.0, 1.0], size=(REPS, len(d)))
    obs = d.mean()
    lo = float((1 + np.sum(((d + bound) * sign).mean(axis=1) <= -abs(obs + bound) + 1e-12))
               / (REPS + 1))
    hi = float((1 + np.sum(((d - bound) * sign).mean(axis=1) >= abs(obs - bound) - 1e-12))
               / (REPS + 1))
    return {"bound": float(bound), "n_nations": n, "delta": float(obs),
            "p_lower": lo, "p_upper": hi, "equivalent": bool(max(lo, hi) < 0.05)}


def block_contrast(a, b):
    """Regime contrast at the block level, the unit the example draw varies over."""
    ga = collections.defaultdict(list)
    gb = collections.defaultdict(list)
    for r in a:
        ga[r["block"]].append(r)
    for r in b:
        gb[r["block"]].append(r)
    blocks = sorted(set(ga) & set(gb))
    d = [MACRO([r["gold"] for r in ga[k]], [r["pred"] for r in ga[k]])
         - MACRO([r["gold"] for r in gb[k]], [r["pred"] for r in gb[k]]) for k in blocks]
    fav = sum(x < 0 for x in d)
    return {"blocks": blocks, "delta": d, "mean": float(np.mean(d)), "n_blocks": len(d),
            "n_favour_heldout": fav, "p_sign": 0.5 ** len(d) if fav == len(d) else None,
            "floor": 3 / (2 ** len(d) + 1)}


def mcnemar(a, b):
    """Paired item-level disagreement. Valid for these example sets, not beyond them."""
    ca = {r["provision_id"]: r["gold"] == r["pred"] for r in a}
    cb = {r["provision_id"]: r["gold"] == r["pred"] for r in b}
    both = sorted(set(ca) & set(cb))
    x = sum(ca[k] and not cb[k] for k in both)
    y = sum(cb[k] and not ca[k] for k in both)
    n = x + y
    tail = sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n if n else 1.0
    return {"n_paired": len(both), "b_random": x, "b_heldout": y,
            "p": float(min(1.0, 2 * tail))}


def main(pkg, out):
    pkg = pathlib.Path(pkg)
    idx = json.load(open(pkg / "index.json"))
    labels = set(idx["labels"])
    model, effort = "unknown", None
    mf = pkg / "replies/model.txt"
    if mf.exists():
        for l in mf.read_text().splitlines():
            if l.lower().startswith("model:"):
                model = l.split(":", 1)[1].strip()
            elif l.lower().startswith("reasoning effort:"):
                effort = l.split(":", 1)[1].strip()

    arms, unparsed, missing = collections.defaultdict(list), {}, {}
    for tag, b in idx["blocks"].items():
        f = pkg / "replies" / f"{tag}.txt"
        if not f.exists():
            print(f"MISSING reply: {tag}")
            continue
        got, bad = parse(f.read_text(), len(b["items"]), labels)
        unparsed[tag], missing[tag] = bad, len(b["items"]) - len(got)
        for i, (pid, g, nat) in enumerate(zip(b["items"], b["gold"], b["nation"]), 1):
            if i in got:
                arms[b["regime"]].append({"provision_id": pid, "nation": nat, "block": b["block"],
                                          "gold": g, "pred": got[i]})
        print(f"{tag:22s} parsed {len(got):3d}/{len(b['items']):3d}  "
              f"unparseable lines {bad}")

    if len(arms) < 2:
        raise SystemExit("need both regimes before scoring")

    res = {"model": model, "effort": effort, "k_examples": idx["k_examples"],
           "unparseable_lines": unparsed, "unlabelled_items": missing, "arms": {}}
    pn = {}
    for reg, items in arms.items():
        pn[reg] = per_nation(items)
        res["arms"][reg] = {"n": len(items), "n_nations": len(pn[reg]),
                            "macro_f1": cluster_ci(items)}
        v = res["arms"][reg]["macro_f1"]
        print(f"\n{reg:12s} n={len(items):3d}  macro-F1 {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]")

    a, b = "random", "doc_heldout"
    if a in pn and b in pn:
        d, p, n, _, _ = signflip(pn[a], pn[b])
        rand = res["arms"][a]["macro_f1"][0]
        res["gap"] = {"delta": d, "p": p, "n_nations": n,
                      "ratio": rand / res["arms"][b]["macro_f1"][0]
                      if res["arms"][b]["macro_f1"][0] else None}
        res["gap"]["note"] = ("Nation-level pairing; the example set is fixed within a block, "
                              "so this p overstates the replication available")
        print(f"\nrandom - doc-held-out = {d:+.3f}  ratio {res['gap']['ratio']:.2f}x")
        res["by_block"] = block_contrast(arms[a], arms[b])
        bb = res["by_block"]
        print(f"  per block {[round(x, 3) for x in bb['delta']]}  mean {bb['mean']:+.4f}")
        print(f"  {bb['n_favour_heldout']}/{bb['n_blocks']} blocks favour the hold-out, "
              f"one-sided sign test p = {bb['p_sign']:.4f}")
        print(f"  the {bb['n_blocks']}-block permutation floor is {bb['floor']:.3f}, so this design "
              f"bounds the direction of the gap but not its size")
        res["mcnemar"] = mcnemar(arms[a], arms[b])
        m = res["mcnemar"]
        print(f"  item level: {m['b_random']} items only the random arm gets right, "
              f"{m['b_heldout']} only the hold-out arm, exact p = {m['p']:.5f}")

    json.dump(res, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default="/Users/austincheng/Downloads/llm-run")
    ap.add_argument("--out", default=str(ROOT / "results/llm_regimes.json"))
    a = ap.parse_args()
    main(a.pkg, a.out)
