"""Polite HTTP fetcher with on-disk cache and robots.txt enforcement."""
import hashlib, pathlib, time, threading, urllib.parse, urllib.robotparser
import requests

UA = "TribalLegalNLP-research/1.0 (academic corpus study; contact via repository)"
CACHE = pathlib.Path(__file__).resolve().parents[1] / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

_robots, _last, _lock = {}, {}, threading.Lock()
MIN_INTERVAL = 0.5


def allowed(url):
    p = urllib.parse.urlparse(url)
    root = f"{p.scheme}://{p.netloc}"
    if root not in _robots:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(root + "/robots.txt")
        try:
            rp.read()
        except Exception:
            rp = None
        _robots[root] = rp
    rp = _robots[root]
    return True if rp is None else rp.can_fetch(UA, url)


def _throttle(host):
    with _lock:
        wait = MIN_INTERVAL - (time.time() - _last.get(host, 0))
        if wait > 0:
            time.sleep(wait)
        _last[host] = time.time()


def get(url, timeout=30, force=False):
    """Return page text, or None. Caches by URL hash; obeys robots.txt."""
    key = CACHE / (hashlib.sha1(url.encode()).hexdigest() + ".html")
    if key.exists() and not force:
        return key.read_text(encoding="utf-8", errors="replace")
    if not allowed(url):
        return None
    _throttle(urllib.parse.urlparse(url).netloc)
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    r.encoding = r.encoding or "utf-8"
    key.write_text(r.text, encoding="utf-8")
    return r.text
