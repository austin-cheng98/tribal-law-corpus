"""Does criminal law converge across Nations more than civil procedure?

Divergence is measured between Nations within each domain. Cells are subsampled to a
common size before every comparison, because Jensen-Shannon divergence estimated from
small samples is biased upward and cell sizes differ by an order of magnitude.
"""
import collections, itertools, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import SEED, corpus_path

ROOT = pathlib.Path(__file__).resolve().parents[1]
LABELS = ["DEFINITION", "GRANT_OF_AUTHORITY", "SANCTION_REMEDY", "RIGHT_ENTITLEMENT",
          "PROHIBITION", "OBLIGATION", "PROCEDURE", "SCOPE_APPLICABILITY"]
MIN_CELL = 20


def dist(items, key, vocab):
    c = collections.Counter(x[key] for x in items if x.get(key))
    v = np.array([c[k] for k in vocab], float) + 0.5      # Jeffreys smoothing
    return v / v.sum()


def jsd(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def cell_divergence(cells, nations, key, vocab, rng, n_draw, reps=200):
    """Mean pairwise JSD between Nations, averaged over equal-size subsamples."""
    per_pair = collections.defaultdict(list)
    for _ in range(reps):
        d = {}
        for n in nations:
            pool = cells[n]
            idx = rng.choice(len(pool), n_draw, replace=False)
            d[n] = dist([pool[i] for i in idx], key, vocab)
        for a, b in itertools.combinations(nations, 2):
            per_pair[(a, b)].append(jsd(d[a], d[b]))
    return {k: float(np.mean(v)) for k, v in per_pair.items()}


def analyse(rows, key, vocab, out):
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r.get(key):
            cells[r["domain"]][r["nation"]].append(r)
    nations = sorted(set(cells["criminal"]) & set(cells["civil"]))
    nations = [n for n in nations
               if len(cells["criminal"][n]) >= MIN_CELL and len(cells["civil"][n]) >= MIN_CELL]
    n_draw = min(min(len(cells[d][n]) for n in nations) for d in ("criminal", "civil"))
    rng = np.random.default_rng(SEED)
    print(f"  {len(nations)} Nations crossed, subsample n={n_draw} per cell", flush=True)
    pc = cell_divergence(cells["criminal"], nations, key, vocab, rng, n_draw)
    pv = cell_divergence(cells["civil"], nations, key, vocab, rng, n_draw)
    pairs = sorted(pc)
    dc = np.array([pc[p] for p in pairs])
    dv = np.array([pv[p] for p in pairs])
    diff = float(dv.mean() - dc.mean())

    # Sign-flip over Nations, not pairs: each Nation appears in n-1 pairs, so permuting
    # pairs treats dependent quantities as exchangeable and understates p.
    pd = {n: np.mean([pv[p] - pc[p] for p in pairs if n in p]) for n in nations}
    d = np.array([pd[n] for n in nations])
    signs = np.array(list(itertools.product([-1, 1], repeat=len(d))))   # exact enumeration
    null = (signs * d).mean(axis=1)
    p_perm = float(np.mean(np.abs(null) >= abs(d.mean()) - 1e-12))
    p_floor = 2.0 / 2 ** len(nations)      # smallest two-sided p this many Nations can give

    # Nation-level bootstrap: resample Nations, not pairs, to respect clustering
    rng3 = np.random.default_rng(SEED + 2)
    boot = []
    for _ in range(5000):
        s = rng3.choice(nations, len(nations), replace=True)
        sel = [(a, b) for a, b in pairs if a in s and b in s]
        if len(sel) < 3:
            continue
        boot.append(np.mean([pv[p] for p in sel]) - np.mean([pc[p] for p in sel]))
    lo, hi = np.quantile(boot, [0.025, 0.975])

    res = {"feature": key, "n_nations": len(nations), "subsample_n": int(n_draw),
           "mean_jsd_criminal": float(dc.mean()), "mean_jsd_civil": float(dv.mean()),
           "civil_minus_criminal": diff, "p_signflip": p_perm,
           "p_floor": float(p_floor), "n_nations_signflip": len(nations),
           "boot_ci": [float(lo), float(hi)], "n_pairs": len(pairs),
           "nations": nations}
    print(f"  JSD criminal {dc.mean():.4f} | civil {dv.mean():.4f} | "
          f"diff {diff:+.4f} [{lo:+.4f},{hi:+.4f}] p={p_perm:.4f} "
          f"(floor {p_floor:.4f}, {len(nations)} Nations)", flush=True)
    out.append(res)
    return res


OK_CLUSTER = {"Pawnee Nation of Oklahoma", "Sac & Fox Nation", "Kickapoo Tribe of Oklahoma",
              "Wyandotte Nation", "Iowa Tribe of Oklahoma", "Delaware Tribe of Indians"}


def main():
    base = [json.loads(l) for l in open(corpus_path())]
    base = [r for r in base if r.get("domain")]
    for r in base:
        m = r.get("modality") or []
        r["modality_1"] = m[0] if m else None
    conds = [("all provisions", base),
             ("near-duplicates removed", [r for r in base if not r.get("xnation_dup")]),
             ("Oklahoma cluster removed", [r for r in base if r["nation"] not in OK_CLUSTER])]
    out = []
    for name, rows in conds:
        print(f"\n== {name} ({len(rows)} provisions) ==")
        for feat, vocab in (("silver", LABELS), ("modality_1", ["MUST", "MAY", "MUST_NOT", "NONE"])):
            print(f" {feat}:")
            try:
                r = analyse(rows, feat, vocab, out)
                r["condition"] = name
            except (ValueError, KeyError) as e:
                print(f"   insufficient crossed cells ({e})")
    json.dump(out, open(ROOT / "results/comparative.json", "w"), indent=1)


if __name__ == "__main__":
    (ROOT / "results").mkdir(exist_ok=True)
    main()
