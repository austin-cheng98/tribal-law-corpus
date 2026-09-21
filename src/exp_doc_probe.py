"""Document-identification probe: how strong is the source-document fingerprint?

If documents are individually recognisable, a random split over provisions lets a Nation
classifier reach the Nation label through the document rather than through drafting style.
This probe measures that fingerprint directly, and asks how much of it survives when the
Nation label is held constant.
"""
import collections, json, pathlib, sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from data import load, bootstrap_ci, SEED
import masking

ROOT = pathlib.Path(__file__).resolve().parents[1]
MIN_DOC, K = 20, 4


def probe(rows, y, level, k=K):
    folds = StratifiedKFold(k, shuffle=True, random_state=17).split(np.zeros(len(y)), y)
    pred = np.empty(len(y), dtype=object)
    for tr, te in folds:
        cues = set()
        if level >= 3:
            fitted = masking.fit_discriminative_terms([rows[i] for i in tr])
            cues = set().union(*fitted.values()) if fitted else set()
        m = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2)),
            LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
        m.fit([masking.apply(rows[i]["text"], level, cues) for i in tr], y[tr])
        pred[te] = m.predict([masking.apply(rows[i]["text"], level, cues) for i in te])
    return {"accuracy": bootstrap_ci(accuracy_score, y, pred),
            "macro_f1": bootstrap_ci(
                lambda a, b: f1_score(a, b, average="macro", zero_division=0), y, pred)}


def main():
    rows = load()
    big = {d for d, c in collections.Counter(r["doc_id"] for r in rows).items() if c >= MIN_DOC}
    rows = [r for r in rows if r["doc_id"] in big]
    y = np.array([r["doc_id"] for r in rows])
    print(f"{len(rows)} provisions, {len(big)} documents with >={MIN_DOC} provisions, "
          f"{len({r['nation'] for r in rows})} Nations", flush=True)

    out = {"n": len(rows), "n_docs": len(big), "levels": {}}
    for level in (0, 2, 3):
        r = probe(rows, y, level)
        out["levels"][f"M{level}"] = r
        print(f"  document ID  M{level}  acc {r['accuracy'][0]:.3f} "
              f"[{r['accuracy'][1]:.3f},{r['accuracy'][2]:.3f}]  "
              f"macroF1 {r['macro_f1'][0]:.3f} "
              f"[{r['macro_f1'][1]:.3f},{r['macro_f1'][2]:.3f}]", flush=True)

    # within one Nation the label carries no jurisdictional information at all
    per = {}
    bynat = collections.defaultdict(list)
    for r in rows:
        bynat[r["nation"]].append(r)
    for nat, rs in sorted(bynat.items()):
        docs = collections.Counter(r["doc_id"] for r in rs)
        if len(docs) < 2 or min(docs.values()) < MIN_DOC:
            continue
        yy = np.array([r["doc_id"] for r in rs])
        per[nat] = probe(rs, yy, 0)["macro_f1"][0]
        print(f"    within {nat[:32]:32s} {len(docs)} docs  macroF1 {per[nat]:.3f}", flush=True)
    out["within_nation"] = per
    if per:
        print(f"  within-Nation mean macro-F1 {np.mean(list(per.values())):.3f}")
    json.dump(out, open(ROOT / "results/doc_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
