"""How often do the document-held-out and leave-one-Nation-out training sets coincide?

On the verified subset each Nation contributes few provisions per document, so the random
test sample within a Nation usually touches every document that Nation has. Holding those
documents out then removes the Nation entirely and the two regimes become one experiment.
This counts the folds where that happens, which qualifies how Table 3 should be read.
"""
import collections, json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from exp_classify import SEED, regimes
from data import corpus_path

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST_FRAC, MIN_TEST = 0.5, 10


def main():
    rows = [json.loads(l) for l in open(corpus_path())]
    g = json.load(open(ROOT / "results/gold.json"))["gold"]
    gold = [dict(r, gold=g[r["provision_id"]]) for r in rows if g.get(r["provision_id"])]

    bynat = collections.defaultdict(list)
    for r in gold:
        bynat[r["nation"]].append(r)
    sp = np.random.default_rng(SEED)
    cut = {}
    for n, v in bynat.items():
        v = [v[i] for i in sp.permutation(len(v))]
        cut[n] = v[:max(MIN_TEST, int(round(TEST_FRAC * len(v))))]
    folds = {n: v for n, v in sorted(cut.items()) if len(v) >= MIN_TEST}

    rng = np.random.default_rng(SEED)
    same = []
    for n, test in folds.items():
        r = regimes(gold, n, {x["provision_id"] for x in test},
                    {x["doc_id"] for x in test}, rng)
        a = {x["provision_id"] for x in r["R1d_doc_heldout"]}
        b = {x["provision_id"] for x in r["R2_lono"]}
        if a == b:
            same.append(n)
    out = {"n_folds": len(folds), "n_identical": len(same), "identical_nations": same}
    print(f"R1d and R2 train on identical rows in {len(same)} of {len(folds)} folds")
    json.dump(out, open(ROOT / "results/regime_overlap.json", "w"), indent=1)


if __name__ == "__main__":
    main()
