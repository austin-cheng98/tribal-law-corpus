"""Replicate the source-diversity manipulation outside law, on public non-legal corpora.

Each corpus has the same nesting as the state codes: passages sit inside a source
document, the document carries one value of the label, and nesting is therefore fixed
at 1.0 by construction. Only the number of documents per label value is varied, with
passages per value held constant, so the descent cannot be attributed to nesting.
"""
import collections
import os, io, json, re, pathlib, time, urllib.request

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

SEED = 20260101
OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(parents=True, exist_ok=True)

KS = [2, 3, 4, 6, 10, 16, 25, 40]   # documents per label value
MINSEG = 30            # a document counts only with this many usable passages
TARGET = 2 * MINSEG    # passages per label value, identical at every k
MAX_VALUES = 20        # label values retained, as in the state codes
REPEATS = 10
N_PERM = 5
MINW, MAXW = 25, 400   # passage length window, as in the state-code run
DEADLINE = 10.0 * 3600
T0 = time.time()
MACRO = lambda a, b: f1_score(a, b, average="macro", zero_division=0)
NUM = re.compile(r"\b\d[\d,.\-]*\b")


def log(*a):
    print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)


def window(text, minseg, minw, maxw):
    """Passages from a document: blank-line paragraphs, then lines, then word windows."""
    for sep in (r"\n\s*\n+", r"\n+"):
        out = []
        for p in re.split(sep, text):
            p = " ".join(p.split())
            if minw <= len(p.split()) <= maxw:
                out.append(p)
        if len(out) >= minseg:
            return out
    w = text.split()
    size = max(minw, 60)
    out = [" ".join(w[i:i + size]) for i in range(0, len(w) - size + 1, size)]
    return out


def probe(X, y, groups, mode, rng, nf=4):
    """Macro-F1 for predicting y, splitting either at random or by source document."""
    idx = np.arange(len(y))
    if mode == "random":
        rng.shuffle(idx)
        folds = np.array_split(idx, nf)
    else:
        gs = sorted(set(groups))
        rng.shuffle(gs)
        chunks = [set(c) for c in np.array_split(np.array(gs, dtype=object), nf)]
        folds = [np.array([i for i in idx if groups[i] in c]) for c in chunks]
    scores = []
    for f in folds:
        te = set(f.tolist())
        tr = np.array([i for i in idx if i not in te])
        if len(tr) == 0 or len(f) == 0 or len(set(y[i] for i in tr)) < 2:
            continue
        m = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2),
                            max_features=200_000),
            LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced"))
        m.fit([X[i] for i in tr], [y[i] for i in tr])
        p = m.predict([X[i] for i in f])
        unseen = len(set(y[i] for i in f) - set(y[i] for i in tr))
        scores.append((MACRO([y[i] for i in f], p), unseen))
    if not scores:
        return [float("nan")] * 2
    return [float(np.mean([s[0] for s in scores])), float(np.mean([s[1] for s in scores]))]


def theil_u(y, g):
    """Share of label entropy the source document resolves, minus its permuted value."""
    def u(lab):
        n = len(lab)
        cy = collections.Counter(lab)
        hy = -sum(c / n * np.log(c / n) for c in cy.values())
        if hy <= 0:
            return 0.0
        cond = collections.defaultdict(collections.Counter)
        for a, b in zip(g, lab):
            cond[a][b] += 1
        hyx = sum(sum(c.values()) / n * -sum(v / sum(c.values()) * np.log(v / sum(c.values()))
                                            for v in c.values()) for c in cond.values())
        return (hy - hyx) / hy
    obs = u(y)
    rng = np.random.default_rng(7)
    null = float(np.mean([u(list(rng.permutation(y))) for _ in range(20)]))
    return float(obs), float((obs - null) / (1 - null)) if null < 1 else float(obs)


def nfolds(k):
    """A fold count above k would empty some value's training set."""
    return min(4, k)


def subsample(pool, k, rng, total):
    """k documents per label value, then `total` passages drawn from their pooled passages."""
    out = []
    for val, docs in pool.items():
        keys = sorted(docs)
        keys = [keys[i] for i in rng.permutation(len(keys))[:k]]
        cand = [(val, d, t) for d in keys for t in docs[d]]
        if len(cand) > total:
            cand = [cand[i] for i in rng.permutation(len(cand))[:total]]
        out += cand
    return out


def run_point(sub, rng, k, maskre, perm=False):
    X = [NUM.sub("[NUM]", maskre.sub("[LABEL]", t)) if maskre else NUM.sub("[NUM]", t)
         for _, _, t in sub]
    y = [v for v, _, _ in sub]
    g = [d for _, d, _ in sub]
    nf = nfolds(k)
    a = probe(X, y, g, "random", rng, nf=nf)
    b = probe(X, y, g, "group", rng, nf=nf)
    uraw, uadj = theil_u(y, g)
    row = {"n_passages": len(sub), "n_docs": len(set(g)), "n_folds": nf,
           "random_f1": a[0], "doc_heldout_f1": b[0],
           "ratio": a[0] / b[0] if b[0] else None,
           "theil_u": uraw, "theil_u_adj": uadj, "unseen_classes_group": b[1]}
    if perm:
        q = [probe(X, list(rng.permutation(y)), g, "group", rng, nf=nf)[0] for _ in range(N_PERM)]
        row["perm_null_f1"] = float(np.mean(q))
    return row


def shape(pool, minseg):
    """Drop thin documents, then fix one value set at the deepest dose the corpus supports."""
    pool = {v: {d: t for d, t in ds.items() if len(t) >= minseg} for v, ds in pool.items()}
    counts = sorted(((len(ds), v) for v, ds in pool.items()), reverse=True)
    log("  doc counts per value:", [(v, c) for c, v in counts[:24]])
    ks = [k for k in KS if sum(1 for c, _ in counts if c >= k) >= 4]
    if not ks:
        return {}, [], counts
    kmax = max(ks)
    keep = [v for c, v in counts if c >= kmax][:MAX_VALUES]
    return {v: pool[v] for v in keep}, ks, counts


# ----------------------------------------------------------------------- loaders

def parquet_files(repo, config=None, split="train"):
    """Auto-converted parquet shards, so a dataset with a loading script still opens."""
    url = f"https://huggingface.co/api/datasets/{repo}/parquet"
    with urllib.request.urlopen(url, timeout=120) as r:
        tree = json.load(io.TextIOWrapper(r, "utf-8"))
    cfg = config if config in tree else ("default" if "default" in tree else sorted(tree)[0])
    sp = split if split in tree[cfg] else sorted(tree[cfg])[0]
    log("  ", repo, "config", cfg, "split", sp, "of", sorted(tree))
    return cfg, sp, tree[cfg][sp]


def rows_parquet(paths):
    """Read shards one at a time off local disk, so no shard is held in memory twice."""
    import pyarrow.parquet as pq
    for i, path in enumerate(paths):
        f = pq.ParquetFile(path)
        log(f"    shard {i}: {f.metadata.num_rows} rows, columns {f.schema_arrow.names}")
        for batch in f.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                yield row
        os.remove(path)


def fetch_shards(repo, cfg, sp, urls):
    """Pull the shards to disk. The hub client first, a plain download as the fallback."""
    try:
        from huggingface_hub import hf_hub_download
        names = [f"{cfg}/{sp}/{n}.parquet" for n in range(len(urls))]
        out = [hf_hub_download(repo_id=repo, filename=n, repo_type="dataset",
                               revision="refs/convert/parquet") for n in names]
        log("    hf_hub_download ok,", len(out), "shards")
        return out
    except Exception as e:
        log("    hf_hub_download failed", repr(e)[:160])
    out = []
    for i, u in enumerate(urls):
        full = u if u.startswith("http") else "https://huggingface.co/" + u.lstrip("/")
        dest = f"/tmp/{repo.replace('/', '_')}_{i}.parquet"
        urllib.request.urlretrieve(full, dest)
        out.append(dest)
    log("    direct download ok,", len(out), "shards")
    return out


def stream(repo, config=None, split="train"):
    """Row iterator. Tries the plain loader, then the converted parquet shards."""
    from datasets import load_dataset
    try:
        d = load_dataset(repo, config, split=split, streaming=True)
        log("    load_dataset streaming ok")
        return d
    except Exception as e:
        log("    load_dataset failed", repr(e)[:160])
    cfg, sp, urls = parquet_files(repo, config, split)
    log("  ", repo, len(urls), "parquet shards")
    return rows_parquet(fetch_shards(repo, cfg, sp, urls))


def harvest(rows, label_of, doc_of, text_of, spec, cap_rows, per_value):
    """Fill a value -> document -> passages pool, stopping once every value is full."""
    pool = collections.defaultdict(lambda: collections.defaultdict(list))
    full, n = set(), 0
    for r in rows:
        n += 1
        if n % 20_000 == 0:
            log(f"   {n} rows, {len(pool)} values, "
                f"{sum(len(d) for d in pool.values())} documents, {len(full)} full")
        if n >= cap_rows or time.time() - T0 > DEADLINE:
            break
        v = label_of(r)
        d = (doc_of(r) if doc_of else None) or f"row{n}"
        if not v or v in full:
            continue
        segs = window(text_of(r) or "", spec["minseg"], spec["minw"], spec["maxw"])
        if len(segs) < spec["minseg"]:
            continue
        pool[v][d] = segs[:spec["minseg"] * 2]
        if len(pool[v]) >= per_value:
            full.add(v)
    log(f"   read {n} rows -> {len(pool)} values, "
        f"{sum(len(d) for d in pool.values())} documents")
    return pool


def load_arxiv(spec):
    """Paragraphs nested in papers nested in the arXiv primary category."""
    rows = stream("ccdv/arxiv-classification", "no_ref", "train")
    pool = harvest(rows, lambda r: str(r.get("label")), None,
                   lambda r: r.get("text"), spec, 40_000, spec["per_value"])
    return pool, re.compile(r"\barxiv\b", re.I)


def load_patent(spec):
    """Paragraphs nested in patents nested in the IPC section the office assigned."""
    rows = stream("ccdv/patent-classification", "patent", "train")
    pool = harvest(rows, lambda r: str(r.get("label")), None,
                   lambda r: r.get("text"), spec, 40_000, spec["per_value"])
    return pool, None


def load_gutenberg(spec):
    """Paragraphs nested in books nested in the author who wrote them."""
    rows = stream("sedthh/gutenberg_english", None, "train")
    names, seen = set(), []

    def author(r):
        # the schema is not documented, so report the first row and read the
        # author from whichever field actually carries it
        if not seen:
            seen.append(1)
            log("    gutenberg row keys:", sorted(r.keys()))
            for k, v in r.items():
                log(f"      {k}: {str(v)[:220]!r}")
        m = r.get("METADATA")
        if isinstance(m, str):
            try:
                m = json.loads(m or "{}")
            except Exception:
                m = {}
        if not isinstance(m, dict):
            m = {}
        a = (m.get("authors") or m.get("author") or m.get("Authors")
             or r.get("authors") or r.get("author"))
        if isinstance(a, (list, tuple)):
            a = a[0] if a else None
        a = " ".join(str(a or "").split())
        if not a or len(a) < 4 or a.lower() in ("none", "anonymous", "various"):
            return None
        names.add(a)
        return a

    pool = harvest(rows, author, lambda r: str(r.get("id", ""))[:64] or None,
                   lambda r: r.get("TEXT") or r.get("text"), spec, 60_000, spec["per_value"])
    toks = sorted({w for a in names for w in re.findall(r"[A-Za-z]{4,}", a)})
    mask = re.compile(r"\b(" + "|".join(re.escape(t) for t in toks[:4000]) + r")\b", re.I) if toks else None
    return pool, mask


def load_news(spec):
    """Paragraphs nested in articles nested in the outlet that published them."""
    rows = stream("vblagoje/cc_news", None, "train")
    doms = set()

    def dom(r):
        d = (r.get("domain") or "").lower().strip()
        if not d or "." not in d:
            return None
        doms.add(d)
        return d

    pool = harvest(rows, dom, lambda r: (r.get("url") or r.get("title") or "")[:200] or None,
                   lambda r: r.get("text"), spec, 400_000, spec["per_value"])
    toks = sorted({w for d in doms for w in re.findall(r"[A-Za-z]{4,}", d)})
    mask = re.compile(r"\b(" + "|".join(re.escape(t) for t in toks[:4000]) + r")\b", re.I) if toks else None
    return pool, mask


CORPORA = [
    ("arxiv_categories", "arXiv primary category", "paper", "paragraph", load_arxiv,
     dict(minseg=30, minw=25, maxw=400, per_value=90)),
    ("patent_classes", "IPC section", "patent", "paragraph", load_patent,
     dict(minseg=30, minw=25, maxw=400, per_value=90)),
    ("gutenberg_authors", "author", "book", "paragraph", load_gutenberg,
     dict(minseg=30, minw=25, maxw=400, per_value=60)),
    ("news_outlets", "publishing outlet", "article", "paragraph", load_news,
     dict(minseg=12, minw=15, maxw=400, per_value=90)),
]


def main():
    res = {"seed": SEED, "dose_levels": KS, "min_passages_per_doc": MINSEG,
           "passages_per_value": TARGET, "max_values": MAX_VALUES,
           "repeats": REPEATS, "corpora": {}}
    path = OUT / "external_dose.json"

    def save():
        path.write_text(json.dumps(res, indent=1))

    for name, lab, doc, seg, loader, spec in CORPORA:
        log("=" * 20, name)
        rec = {"label": lab, "document": doc, "passage": seg, "spec": spec, "fixed_n": []}
        res["corpora"][name] = rec
        save()
        try:
            raw, maskre = loader(spec)
        except Exception as e:
            rec["error"] = repr(e)[:300]
            log(name, "LOAD FAILED", rec["error"])
            save()
            continue
        pool, ks, counts = shape(raw, spec["minseg"])
        rec["doc_counts"] = [[v, c] for c, v in counts[:40]]
        if not pool:
            rec["error"] = "no dose level keeps 4 label values"
            log(name, rec["error"])
            save()
            continue
        rec.update(n_values=len(pool), values=sorted(pool), dose_levels_run=ks,
                   docs_available=[len(pool[v]) for v in sorted(pool)])
        log(name, f"{len(pool)} values, ladder {ks}")
        save()
        rng = np.random.default_rng(SEED)
        for k in ks:
            if time.time() - T0 > DEADLINE:
                break
            reps = [run_point(subsample(pool, k, rng, TARGET), rng, k, maskre, perm=(r == 0))
                    for r in range(REPEATS)]
            draws = [r["ratio"] for r in reps if r["ratio"]]
            row = {"k": k, "n_passages": reps[0]["n_passages"], "n_values": len(pool),
                   "n_folds": reps[0]["n_folds"], "ratio_draws": draws,
                   "ratio_mean": float(np.mean(draws)) if draws else None,
                   "repeats": len(reps),
                   "random_f1": float(np.mean([r["random_f1"] for r in reps])),
                   "doc_heldout_f1": float(np.mean([r["doc_heldout_f1"] for r in reps])),
                   "theil_u_adj": float(np.mean([r["theil_u_adj"] for r in reps])),
                   "unseen_classes_group": float(np.mean([r["unseen_classes_group"] for r in reps])),
                   "perm_null_f1": reps[0].get("perm_null_f1")}
            row["ratio"] = (row["random_f1"] / row["doc_heldout_f1"]
                            if row["doc_heldout_f1"] else None)
            rec["fixed_n"].append(row)
            save()
            log(name, f"k={k:3d} n={row['n_passages']:5d} rand {row['random_f1']:.3f} "
                      f"doc {row['doc_heldout_f1']:.3f} ratio {row['ratio']:.2f} "
                      f"U {row['theil_u_adj']:.3f} null {row['perm_null_f1']}")
    log("wrote external_dose.json")


if __name__ == "__main__":
    main()
