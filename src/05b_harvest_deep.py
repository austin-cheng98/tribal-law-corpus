"""Second harvest pass: multiple seeds per Nation, depth 3, wider page budget.
Documents already on disk are not refetched."""
import json, pathlib, sys, time, urllib.parse, collections
import concurrent.futures as cf
import requests
from bs4 import BeautifulSoup
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import importlib.util
_s = importlib.util.spec_from_file_location("h", pathlib.Path(__file__).parent / "05_harvest.py")
H = importlib.util.module_from_spec(_s); _s.loader.exec_module(H)

ROOT = pathlib.Path(__file__).resolve().parents[1]
(ROOT / "data/interim").mkdir(parents=True, exist_ok=True)
RAW = ROOT / "data/raw/docs"
RAW.mkdir(parents=True, exist_ok=True)


def harvest_multi(item, max_pages=240, max_depth=3):
    nation, urls = item
    q = collections.deque((u, 0, "") for u in urls)
    hosts = {urllib.parse.urlparse(u).netloc for u in urls}
    seen, kept, pages = set(urls), [], 0
    while q and pages < max_pages:
        u, d, anchor = q.popleft()
        fp_h, fp_p = RAW / f"{H.key(u)}.html", RAW / f"{H.key(u)}.pdf"
        if fp_h.exists() or fp_p.exists():
            continue
        r = H.grab(u)
        pages += 1
        time.sleep(0.2)
        if r is None:
            continue
        ct = r.headers.get("content-type", "").lower()
        if "pdf" in ct or u.lower().endswith(".pdf"):
            if H.TARGET.search(anchor + " " + u) and len(r.content) > 8000:
                fp_p.write_bytes(r.content)
                kept.append({"nation": nation, "url": r.url, "anchor": anchor[:200],
                             "file": fp_p.name, "kind": "pdf", "bytes": len(r.content)})
            continue
        soup = BeautifulSoup(r.text, "lxml")
        text = soup.get_text(" ", strip=True)
        if H.TARGET.search(anchor + " " + u + " " + text[:400]) and len(text) > 1800:
            fp_h.write_text(r.text, encoding="utf-8")
            kept.append({"nation": nation, "url": r.url, "anchor": anchor[:200],
                         "file": fp_h.name, "kind": "html", "chars": len(text)})
        if d >= max_depth:
            continue
        for a in soup.find_all("a", href=True):
            nu = urllib.parse.urljoin(r.url, a["href"]).split("#")[0]
            if nu in seen or urllib.parse.urlparse(nu).netloc not in hosts or H.SKIP.search(nu):
                continue
            t = a.get_text(" ", strip=True)
            if H.FOLLOW.search(t) or H.FOLLOW.search(nu):
                seen.add(nu)
                q.append((nu, d + 1, t))
    return {"nation": nation, "seed": urls[0], "pages_fetched": pages, "docs": kept}


def main():
    seeds = json.load(open(ROOT / "data/interim/seeds_deep.json"))
    out = []
    with cf.ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(harvest_multi, it): it[0] for it in seeds.items()}
        for i, f in enumerate(cf.as_completed(futs), 1):
            try:
                r = f.result()
            except Exception:
                continue
            out.append(r)
            if r["docs"]:
                print(f"  [{i:3d}/{len(seeds)}] +{len(r['docs']):3d} docs  {r['nation'][:48]}", flush=True)
            if i % 40 == 0:
                print(f"  ... {i}/{len(seeds)}, running total {sum(len(x['docs']) for x in out)}", flush=True)
    json.dump(out, open(ROOT / "data/interim/harvest_deep.json", "w"), indent=1)
    print(f"\nNEW {sum(len(r['docs']) for r in out)} documents")


if __name__ == "__main__":
    main()
