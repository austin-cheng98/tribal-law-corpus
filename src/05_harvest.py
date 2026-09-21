"""Depth-limited, same-domain BFS over Nation code sites; store procedure documents (HTML/PDF)."""
import hashlib, json, pathlib, re, sys, time, urllib.parse, collections
import concurrent.futures as cf
import requests
from bs4 import BeautifulSoup
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from fetch import UA, allowed

ROOT = pathlib.Path(__file__).resolve().parents[1]
(ROOT / "data/interim").mkdir(parents=True, exist_ok=True)
RAW = ROOT / "data/raw/docs"
RAW.mkdir(parents=True, exist_ok=True)
HDR = {"User-Agent": UA}

# Link text or URL that plausibly leads to procedure law.
FOLLOW = re.compile(
    r"law\s*(and|&)?\s*order|code|title\s+[ivxlcdm0-9]+|chapter|ordinance|rules?\b|procedure"
    r"|criminal|civil|court|judicial|justice|statut", re.I)
# Documents we actually want.
TARGET = re.compile(
    r"criminal\s+procedure|rules?\s+of\s+criminal|civil\s+procedure|rules?\s+of\s+civil"
    r"|criminal\s+(code|actions?|offenses?)|civil\s+(code|actions?)|law\s*(and|&)\s*order"
    r"|rules?\s+of\s+court|judicial\s+code|code\s+of\s+justice|courts?\s+and\s+procedure"
    r"|arrest|arraign|sentenc|bail|pleading|summons|service\s+of\s+process|small\s+claims", re.I)
SKIP = re.compile(r"\.(jpg|jpeg|png|gif|svg|zip|docx?|xlsx?|pptx?|mp4|mp3)$|"
                  r"facebook|twitter|instagram|youtube|linkedin|mailto:|tel:|javascript:|#$", re.I)


def key(u):
    return hashlib.sha1(u.encode()).hexdigest()[:16]


def grab(u, timeout=30):
    if not allowed(u) or SKIP.search(u):
        return None
    try:
        r = requests.get(u, headers=HDR, timeout=timeout, allow_redirects=True)
    except Exception:
        return None
    if r.status_code != 200 or "Just a moment" in r.text[:1500]:
        return None
    return r


def harvest(seed_item, max_pages=160, max_depth=2):
    nation, seed = seed_item
    host = urllib.parse.urlparse(seed).netloc
    q = collections.deque([(seed, 0, "")])
    seen, kept = {seed}, []
    pages = 0
    while q and pages < max_pages:
        u, d, anchor = q.popleft()
        r = grab(u)
        pages += 1
        time.sleep(0.25)
        if r is None:
            continue
        ct = r.headers.get("content-type", "").lower()
        if "pdf" in ct or u.lower().endswith(".pdf"):
            if TARGET.search(anchor + " " + u) and len(r.content) > 8000:
                p = RAW / f"{key(u)}.pdf"
                p.write_bytes(r.content)
                kept.append({"nation": nation, "url": r.url, "anchor": anchor[:200],
                             "file": p.name, "kind": "pdf", "bytes": len(r.content)})
            continue
        soup = BeautifulSoup(r.text, "lxml")
        text = soup.get_text(" ", strip=True)
        # Keep HTML pages that look like actual code text, not just menus.
        if TARGET.search(anchor + " " + u + " " + text[:400]) and len(text) > 2500:
            p = RAW / f"{key(u)}.html"
            p.write_text(r.text, encoding="utf-8")
            kept.append({"nation": nation, "url": r.url, "anchor": anchor[:200],
                         "file": p.name, "kind": "html", "chars": len(text)})
        if d >= max_depth:
            continue
        for a in soup.find_all("a", href=True):
            nu = urllib.parse.urljoin(r.url, a["href"]).split("#")[0]
            if nu in seen or urllib.parse.urlparse(nu).netloc != host or SKIP.search(nu):
                continue
            t = a.get_text(" ", strip=True)
            if FOLLOW.search(t) or FOLLOW.search(nu):
                seen.add(nu)
                q.append((nu, d + 1, t))
    return {"nation": nation, "seed": seed, "pages_fetched": pages, "docs": kept}


def main():
    seeds = json.load(open(ROOT / "data/interim/seeds.json"))
    out = []
    with cf.ThreadPoolExecutor(6) as ex:
        futs = {ex.submit(harvest, (n, u)): n for n, u in seeds.items()}
        for i, f in enumerate(cf.as_completed(futs), 1):
            r = f.result()
            out.append(r)
            print(f"  [{i:3d}/{len(seeds)}] {len(r['docs']):3d} docs / {r['pages_fetched']:3d} pages  "
                  f"{r['nation'][:46]}", flush=True)
    json.dump(out, open(ROOT / "data/interim/harvest.json", "w"), indent=1)
    print(f"\nTOTAL {sum(len(r['docs']) for r in out)} documents from "
          f"{sum(1 for r in out if r['docs'])} Nations")


if __name__ == "__main__":
    main()
