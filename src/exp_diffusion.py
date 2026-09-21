"""Cross-Nation textual reuse: how much procedural text is shared between Nations?

Near-duplicate pairs are evidence of model-code diffusion and are also a leakage risk for
leave-one-Nation-out evaluation, so the rate is reported and the pairs are enumerated.
The criminal/civil contrast is tested within Nation, on the Nations that hold enough of
both domains, so it cannot be produced by differences in what each Nation contributes.
"""
import collections, json, pathlib
from hashlib import blake2b
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
K, THRESH, MIN_CELL = 9, 0.80, 20
SEED = 20260101
OK_CLUSTER = {"Pawnee Nation of Oklahoma", "Sac & Fox Nation", "Kickapoo Tribe of Oklahoma",
              "Wyandotte Nation", "Iowa Tribe of Oklahoma", "Delaware Tribe of Indians"}


def shingles(t, k=K):
    w = t.lower().split()
    return {blake2b(" ".join(w[i:i + k]).encode(), digest_size=8).digest()
            for i in range(max(1, len(w) - k + 1))}


def find_pairs(sh, nat, thresh=THRESH):
    """Blocked all-pairs search: candidates share a shingle, then exact Jaccard decides."""
    inv = collections.defaultdict(list)
    for pid, s in sh.items():
        for g in sorted(s)[:60]:
            inv[g].append(pid)
    seen, pairs = set(), []
    for post in inv.values():
        if not 2 <= len(post) <= 40:
            continue
        for a in range(len(post)):
            for b in range(a + 1, len(post)):
                x, y = post[a], post[b]
                if nat[x] == nat[y] or (x, y) in seen:
                    continue
                seen.add((x, y))
                u = len(sh[x] | sh[y])
                if u and len(sh[x] & sh[y]) / u >= thresh:
                    pairs.append((x, y))
    return pairs


def paired_test(rows, involved, reps=20000, seed=SEED):
    """Within-Nation criminal-vs-civil contrast, sign-flipped over Nations."""
    cell = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        c = cell[(r["nation"], r["domain"])]
        c[1] += 1
        c[0] += r["provision_id"] in involved
    have = sorted({n for n, _ in cell})
    nats = [n for n in have
            if cell[(n, "criminal")][1] >= MIN_CELL and cell[(n, "civil")][1] >= MIN_CELL]
    d = np.array([cell[(n, "civil")][0] / cell[(n, "civil")][1]
                  - cell[(n, "criminal")][0] / cell[(n, "criminal")][1] for n in nats])
    rng = np.random.default_rng(seed)
    obs = d.mean()
    null = (d * rng.choice([-1.0, 1.0], size=(reps, len(d)))).mean(axis=1)
    p = (1 + np.sum(np.abs(null) >= abs(obs))) / (reps + 1)
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(5000)])
    return {"nations": nats, "delta": float(obs), "p": float(p),
            "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "per_nation": {n: [cell[(n, "criminal")][0] / cell[(n, "criminal")][1],
                               cell[(n, "civil")][0] / cell[(n, "civil")][1]] for n in nats}}


def rates(rows, involved):
    tot = collections.Counter(r["domain"] for r in rows)
    hit = collections.Counter(r["domain"] for r in rows if r["provision_id"] in involved)
    return {d: [hit[d], tot[d]] for d in ("criminal", "civil")}


def main():
    rows = [json.loads(l) for l in open(ROOT / "data/processed/corpus.jsonl")]
    sh = {r["provision_id"]: shingles(r["text"]) for r in rows}
    nat = {r["provision_id"]: r["nation"] for r in rows}
    pairs = find_pairs(sh, nat)
    involved = {p for pr in pairs for p in pr}
    npair = collections.Counter(tuple(sorted((nat[x], nat[y]))) for x, y in pairs)

    print(f"{len(pairs)} cross-Nation near-duplicate pairs over {len(involved)} provisions "
          f"({len(involved)/len(rows):.1%} of the corpus)")
    base = rates(rows, involved)
    for d in ("criminal", "civil"):
        a, b = base[d]
        print(f"   {d:9s} {a:5d}/{b:5d} = {a/b:.1%} of that domain")

    t = paired_test(rows, involved)
    print(f"\nwithin-Nation contrast over {len(t['nations'])} Nations holding >={MIN_CELL} "
          f"of both: civil - criminal = {t['delta']:+.3f} "
          f"[{t['ci'][0]:+.3f},{t['ci'][1]:+.3f}]  p={t['p']:.4g}")

    t2 = paired_test([r for r in rows if r["nation"] not in OK_CLUSTER], involved)
    print(f"  outside the Oklahoma cluster ({len(t2['nations'])} Nations): "
          f"{t2['delta']:+.3f} [{t2['ci'][0]:+.3f},{t2['ci'][1]:+.3f}]  p={t2['p']:.4g}")

    noOK = [r for r in rows if r["nation"] not in OK_CLUSTER]
    inv2 = {p for p in involved if nat[p] not in OK_CLUSTER}
    r2 = rates(noOK, inv2)
    print("Oklahoma cluster removed: " + "  ".join(
        f"{d} {a}/{b} = {a/b:.1%}" for d, (a, b) in r2.items()))

    print("\nJaccard threshold sensitivity (share of corpus involved):")
    sens = {}
    for th in (0.70, 0.80, 0.90):
        inv = {p for pr in find_pairs(sh, nat, th) for p in pr}
        rr = rates(rows, inv)
        sens[th] = {"share": len(inv) / len(rows),
                    **{d: rr[d][0] / rr[d][1] for d in rr}}
        print(f"   J>={th:.2f}  overall {sens[th]['share']:.1%}  "
              f"criminal {sens[th]['criminal']:.1%}  civil {sens[th]['civil']:.1%}")

    print("\nmost-connected Nation pairs:")
    for (a, b), c in npair.most_common(12):
        print(f"   {c:4d}  {a[:30]:30s} <-> {b[:30]}")

    json.dump({"n_pairs": len(pairs), "n_provisions": len(involved),
               "share": len(involved) / len(rows), "by_domain": base,
               "by_domain_no_ok": r2, "paired": t, "paired_no_ok": t2, "threshold_sensitivity": sens,
               "nation_pairs": [[a, b, c] for (a, b), c in npair.most_common()]},
              open(ROOT / "results/diffusion.json", "w"), indent=1)


if __name__ == "__main__":
    (ROOT / "results").mkdir(exist_ok=True)
    main()
