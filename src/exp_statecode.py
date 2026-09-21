"""Replicate the split-ladder probe on a public multi-jurisdiction statute corpus.

Jurisdiction identity comes from the record grouping in the source file, never from the
text, so the probe cannot be recovering a text-derived labelling rule.
"""
import collections, json, lzma, re, pathlib, time, urllib.request

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260101
URL = ("https://huggingface.co/datasets/pile-of-law/pile-of-law/resolve/main/"
       "data/validation.state_code.jsonl.xz")
MAX_RECORDS = 400
TITLE = re.compile(r"\n(?=(?:Title|TITLE|Chapter|CHAPTER)\s+[0-9IVXLA-Z][0-9A-Za-z.\-]*\s*[-–—.]\s*\S)")
SECTION = re.compile(r"\n(?=(?:Section|Sec\.|§)\s*[0-9])")
MINW, MAXW = 25, 400
T0 = time.time()

STATES = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
          "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois",
          "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
          "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana",
          "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York",
          "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
          "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
          "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
          "District of Columbia", "Puerto Rico", "Guam"]
DEMONYM = ["Alabaman", "Alaskan", "Arizonan", "Arkansan", "Californian", "Coloradan",
           "Floridian", "Georgian", "Hawaiian", "Idahoan", "Illinoisan", "Indianan",
           "Iowan", "Kansan", "Kentuckian", "Marylander", "Michigander", "Minnesotan",
           "Missourian", "Montanan", "Nebraskan", "Nevadan", "Ohioan", "Oklahoman",
           "Oregonian", "Tennessean", "Texan", "Utahn", "Vermonter", "Virginian",
           "Wisconsinite", "Wyomingite"]
NAMES = re.compile("|".join(sorted(STATES + DEMONYM, key=len, reverse=True)), re.I)
NUM = re.compile(r"\b\d[\d,.\-]*\b")
CITE = re.compile(r"§+\s*[\w.\-]+|\b[Cc]hapter\s+[\w.\-]+|\b[Tt]itle\s+[\w.\-]+")
MACRO = lambda a, b: f1_score(a, b, average="macro", zero_division=0)


def mask(t, level):
    if level >= 1:
        t = NAMES.sub("[STATE]", t)
    if level >= 2:
        t = CITE.sub("[CITE]", NUM.sub("[NUM]", t))
    return t


def stream(url, cap):
    """Yield whole JSON records without holding the decompressed shard in memory."""
    d = lzma.LZMADecompressor()
    buf = b""
    n = 0
    with urllib.request.urlopen(url, timeout=300) as r:
        while n < cap:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            buf += d.decompress(chunk)
            while b"\n" in buf and n < cap:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    n += 1
                    yield json.loads(line)


def segment(text):
    """Split a record into (document, [section]) pairs."""
    out = []
    parts = TITLE.split(text)
    for p in parts:
        head = p[:90].strip().replace("\n", " ")
        secs = [s.strip() for s in SECTION.split(p)[1:]]
        secs = [s for s in secs if MINW <= len(s.split()) <= MAXW]
        if secs:
            out.append((head, secs))
    return out


def probe(X, y, groups, mode, rng, cap=None):
    """Macro-F1 for predicting y, splitting either at random or by group."""
    idx = np.arange(len(y))
    if mode == "random":
        rng.shuffle(idx)
        folds = np.array_split(idx, 4)
    else:
        gs = sorted(set(groups))
        rng.shuffle(gs)
        chunks = [set(c) for c in np.array_split(np.array(gs, dtype=object), 4)]
        folds = [np.array([i for i in idx if groups[i] in c]) for c in chunks]
    scores = []
    for f in folds:
        te = set(f.tolist())
        tr = np.array([i for i in idx if i not in te])
        if len(tr) == 0 or len(f) == 0 or len(set(y[i] for i in tr)) < 2:
            continue
        if cap is not None and len(tr) > cap:
            tr = tr[rng.permutation(len(tr))[:cap]]
        m = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2),
                            max_features=200_000),
            LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced"))
        m.fit([X[i] for i in tr], [y[i] for i in tr])
        p = m.predict([X[i] for i in f])
        unseen = len(set(y[i] for i in f) - set(y[i] for i in tr))
        scores.append((MACRO([y[i] for i in f], p),
                       float(np.mean([y[i] == q for i, q in zip(f, p)])), unseen))
    return ([float(np.mean([s[0] for s in scores])), float(np.mean([s[1] for s in scores])),
             float(np.mean([s[2] for s in scores]))] if scores else [float("nan")] * 3)


KS = [2, 5, 10, 20, 40]     # documents per state on the dose-response curve
MINSEC = 30                 # a document must carry this many usable sections
TARGET = 2 * MINSEC         # sections per state, identical at every k
MIN_DOCS = max(KS)
REPEATS = 3                 # subsample draws averaged at each k
N_PERM = 5
DEADLINE = 9 * 3600


def state_of(txt):
    """Assign a record to a state only when one name clearly dominates."""
    hits = collections.Counter({s: len(re.findall(r"\b" + re.escape(s) + r"\b", txt[:400_000]))
                                for s in STATES})
    top = hits.most_common(2)
    if not top or top[0][1] < 50:
        return None, top
    if len(top) > 1 and top[1][1] * 2 > top[0][1]:
        return None, top
    return top[0][0], top


def subsample(pool, k, rng, total=None):
    """k documents per state; `total` sections per state drawn from their pooled sections.

    Pooling before the draw is what makes the section count identical at every k, so the
    curve varies source diversity alone. total=None keeps every section instead.
    """
    out = []
    for st, docs in pool.items():
        keys = sorted(docs)
        keys = [keys[i] for i in rng.permutation(len(keys))[:k]]
        cand = [(st, d, t) for d in keys for t in docs[d]]
        if total is not None and len(cand) > total:
            cand = [cand[i] for i in rng.permutation(len(cand))[:total]]
        out += cand
    return out


def run_point(sub, rng, perm=False):
    """Random vs document-held-out on one subsample, with state names masked."""
    X = [mask(t, 1) for _, _, t in sub]
    y = [st for st, _, _ in sub]
    g = [d for _, d, _ in sub]
    a = probe(X, y, g, "random", rng)
    b = probe(X, y, g, "group", rng)
    row = {"n_sections": len(sub), "n_docs": len(set(g)),
           "random_f1": a[0], "doc_heldout_f1": b[0],
           "ratio": a[0] / b[0] if b[0] else None,
           "unseen_classes_group": b[2]}
    if perm:
        q = [probe(X, list(rng.permutation(y)), g, "group", rng)[0] for _ in range(N_PERM)]
        row["perm_null_f1"] = float(np.mean(q))
        row["perm_null_sd"] = float(np.std(q))
    return row


def main():
    rng = np.random.default_rng(SEED)
    pool = collections.defaultdict(lambda: collections.defaultdict(list))
    meta = []
    for k, rec in enumerate(stream(URL, MAX_RECORDS)):
        txt = rec.get("text", "")
        st, top = state_of(txt)
        docs = segment(txt)
        meta.append({"record": k, "state": st, "top": top,
                     "docs": len(docs), "sections": sum(len(v) for _, v in docs)})
        print(f"rec {k:3d}  state={st}  {len(docs):4d} docs", flush=True)
        if st is None:
            continue
        for h, secs in docs:
            pool[st][f"{k}::{h}"] += secs

    # a document counts only if it carries MINSEC sections, so that k=2 can still
    # reach TARGET sections and the section count stays identical across the curve
    pool = {s: {d: v for d, v in ds.items() if len(v) >= MINSEC} for s, ds in pool.items()}
    pool = {s: d for s, d in pool.items() if len(d) >= MIN_DOCS}
    print(f"\nstates kept (>={MIN_DOCS} docs of >={MINSEC} sections): {len(pool)}", flush=True)
    for s in sorted(pool):
        print(f"  {s:22s} {len(pool[s]):5d} docs", flush=True)
    json.dump(meta, open(OUT / "records.json", "w"), indent=1)
    if len(pool) < 5:
        print("pool too small; stopping", flush=True)
        return

    out = {"seed": SEED, "target_sections_per_state": TARGET, "min_sections_per_doc": MINSEC,
           "min_docs": MIN_DOCS, "repeats": REPEATS, "masking": "state names and demonyms",
           "states": sorted(pool), "fixed_n": []}

    def save():
        json.dump(out, open(OUT / "statecode_dose.json", "w"), indent=1)

    # primary: section count pinned at TARGET per state, so only source diversity moves
    for k in KS:
        if time.time() - T0 > DEADLINE:
            break
        reps = [run_point(subsample(pool, k, rng, TARGET), rng, perm=(r == 0))
                for r in range(REPEATS)]
        row = {"k": k, "n_sections": reps[0]["n_sections"], "n_states": len(pool),
               "random_f1": float(np.mean([r["random_f1"] for r in reps])),
               "doc_heldout_f1": float(np.mean([r["doc_heldout_f1"] for r in reps])),
               "random_sd": float(np.std([r["random_f1"] for r in reps])),
               "doc_heldout_sd": float(np.std([r["doc_heldout_f1"] for r in reps])),
               "unseen_classes_group": float(np.mean([r["unseen_classes_group"] for r in reps])),
               "perm_null_f1": reps[0]["perm_null_f1"], "perm_null_sd": reps[0]["perm_null_sd"]}
        row["ratio"] = row["random_f1"] / row["doc_heldout_f1"] if row["doc_heldout_f1"] else None
        out["fixed_n"].append(row)
        save()
        print(f"[fixed n] k={k:3d}  n={row['n_sections']:6d}  rand {row['random_f1']:.3f}  "
              f"doc {row['doc_heldout_f1']:.3f}  ratio {row['ratio']:.2f}  "
              f"null {row['perm_null_f1']:.3f}  unseen {row['unseen_classes_group']:.1f}  "
              f"[{time.time()-T0:.0f}s]", flush=True)

    print("\nwrote statecode_dose.json")


if __name__ == "__main__":
    main()
