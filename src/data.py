"""Corpus loading, split construction, and bootstrap utilities shared by all experiments."""
import collections, json, pathlib, random
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEED = 20260101


def corpus_path():
    """The labelled corpus where the pipeline has written one, else the released file."""
    p = ROOT / "data/processed/corpus_labeled.jsonl"
    return p if p.exists() else ROOT / "data/processed/corpus.jsonl"


def load(path=None, drop_xnation_dup=False):
    path = path or corpus_path()
    rows = [json.loads(l) for l in open(path)]
    if drop_xnation_dup:
        rows = [r for r in rows if not r.get("xnation_dup")]
    return rows


def nations(rows, min_n=0):
    c = collections.Counter(r["nation"] for r in rows)
    return sorted(n for n, k in c.items() if k >= min_n)


def stratified_sample(rows, n, keys=("nation", "domain"), seed=SEED):
    """Proportional-with-floor sample so every stratum is represented."""
    rng = random.Random(seed)
    buckets = collections.defaultdict(list)
    for r in rows:
        buckets[tuple(r[k] for k in keys)].append(r)
    for b in buckets.values():
        rng.shuffle(b)
    floor = max(1, n // (len(buckets) * 3))
    out = [r for b in buckets.values() for r in b[:floor]]
    pool = [r for b in buckets.values() for r in b[floor:]]
    rng.shuffle(pool)
    out += pool[:max(0, n - len(out))]
    rng.shuffle(out)
    return out[:n]


def bootstrap_ci(fn, y_true, y_pred, n=2000, alpha=0.05, seed=SEED):
    """Percentile bootstrap CI for a metric over paired (true, pred) arrays."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    k = len(y_true)
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, k, k)
        vals[i] = fn(y_true[idx], y_pred[idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(fn(y_true, y_pred)), float(lo), float(hi)


def paired_bootstrap_p(fn, y_true, pred_a, pred_b, n=5000, seed=SEED):
    """Two-sided paired bootstrap test that metric(a) differs from metric(b)."""
    rng = np.random.default_rng(seed)
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    k = len(y_true)
    obs = fn(y_true, pred_a) - fn(y_true, pred_b)
    cnt = 0
    for _ in range(n):
        idx = rng.integers(0, k, k)
        d = fn(y_true[idx], pred_a[idx]) - fn(y_true[idx], pred_b[idx])
        if abs(d - obs) >= abs(obs):
            cnt += 1
    return float(obs), (cnt + 1) / (n + 1)
