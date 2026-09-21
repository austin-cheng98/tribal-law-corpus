"""Convert harvested documents into structured legal provisions with full metadata."""
import collections, json, pathlib, re, sys, unicodedata
import fitz
from bs4 import BeautifulSoup

ROOT = pathlib.Path(__file__).resolve().parents[1]
(ROOT / "data/interim").mkdir(parents=True, exist_ok=True)
RAW = ROOT / "data/raw/docs"

# Section openers, ordered most- to least-specific.
NUM = r"\d+[A-Za-z]?(?:[-.\u2013]\d+[A-Za-z]?)*"
SEC_PATTERNS = [
    rf"(?:^|\n)[ \t]*(?:Sec(?:tion|t|\.)?|Rule|Art(?:icle|\.)?)[ \t]+({NUM})[ \t]*[.:\u2013-]?[ \t]*",
    rf"(?:^|\n)[ \t]*\u00a7+[ \t]*({NUM})[ \t]*[.:\u2013-]?[ \t]*",
    rf"(?:^|\n)[ \t]*({NUM})[ \t]*[.:\u2013][ \t]+(?=[A-Z])",
]
SEC_RE = re.compile("|".join(SEC_PATTERNS), re.I)
SEC_LINE = re.compile(rf"^[ \t]*(?:\u00a7+[ \t]*|(?:Sec(?:tion|t|\.)?|Rule|Art(?:icle|\.)?|Chapter|Title)[ \t]+){NUM}\b", re.I)
CHAP_RE = re.compile(r"(?:^|\n)[ \t]*(CHAPTER|TITLE|ARTICLE|PART|SUBCHAPTER)[ \t]+"
                     r"((?=[0-9]|[IVXLC]{1,7}\b)[0-9IVXLC]+[A-Za-z]?(?:[-.][0-9A-Za-z]+)*)"
                     r"[ \t]*[.:\u2013-]?[ \t]*([^\n]{0,90})")
HIST_RE = re.compile(r"\[\s*History:?(.{0,220}?)\]", re.S | re.I)
YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-2]\d)\b")
DOTLEAD = re.compile(r"\.{4,}\s*\d+")          # table-of-contents dot leaders
TOC_HDR = re.compile(r"table of contents|index of|contents\s*$", re.I)

CRIM_CUE = re.compile(r"criminal|offense|arrest|arraign|bail|bond|sentenc|prosecut|misdemeanor"
                      r"|law\s*(and|&)\s*order|penal|police|jail|probation", re.I)
CIV_CUE = re.compile(r"civil|complaint|summons|pleading|service of process|small claims|judgment"
                     r"|discovery|injunction|garnish|mandamus|replevin|damages", re.I)


def norm(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    return re.sub(r"[ \t ]+", " ", s)


def pdf_text(path):
    """Return (text, n_pages, ocr_needed) with repeated running heads/feet removed."""
    doc = fitz.open(path)
    pages = [p.get_text("text") for p in doc]
    n = len(pages)
    if n == 0:
        return "", 0, True
    chars = sum(len(p.strip()) for p in pages)
    if chars / max(n, 1) < 120:
        return "", n, True
    counts = collections.Counter()
    for p in pages:
        for ln in {l.strip() for l in p.split("\n") if 3 < len(l.strip()) < 90}:
            counts[re.sub(r"\d+", "#", ln)] += 1
    boiler = {k for k, c in counts.items() if c >= max(3, 0.35 * n)}
    out = []
    for p in pages:
        keep = [l for l in p.split("\n") if re.sub(r"\d+", "#", l.strip()) not in boiler]
        out.append("\n".join(keep))
    return norm("\n".join(out)), n, False


def html_text(path):
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        import trafilatura
        got = trafilatura.extract(raw, include_tables=True, include_comments=False,
                                  favor_recall=True)
        if got and len(got) > 400:
            return norm(got), 0, False
    except Exception:
        pass
    soup = BeautifulSoup(raw, "lxml")
    for t in soup(["script", "style", "nav", "header", "footer", "form", "aside"]):
        t.decompose()
    main = soup.find("main") or soup.find("article") or soup.find(
        attrs={"class": re.compile("content|entry|post|body", re.I)}) or soup.body or soup
    return norm(main.get_text("\n", strip=True)), 0, False


LEVEL = {"TITLE": 0, "SUBCHAPTER": 1, "CHAPTER": 1, "ARTICLE": 2, "PART": 3}


def _continuation(text, end):
    """Heading names are often typeset on the line below the enumerator."""
    tail = text[end:end + 200].split("\n")
    for ln in tail[:3]:
        t = ln.strip(" .:-\u2013")
        if not t:
            continue
        w = t.split()
        if not (1 <= len(w) <= 12) or t.endswith((".", ";", ",")):
            return ""
        if not (t.isupper() or t[:1].isupper()):
            return ""
        return norm(t)
    return ""


def context_map(text):
    """Character offset -> enclosing heading stack (TITLE > CHAPTER > ARTICLE > PART)."""
    out = []
    for m in CHAP_RE.finditer(text):
        head = norm(m.group(3)).strip(" .:-")
        if not head:
            head = _continuation(text, m.end())
        out.append((m.start(), m.group(1).upper(),
                    f"{m.group(1).title()} {m.group(2)}".strip(), head))
    return out


def nearest(marks, pos):
    """Resolve the stack at `pos`; a heading clears every level beneath it."""
    stack = [None] * 4
    for p, kind, lab, head in marks:
        if p > pos:
            break
        lv = LEVEL[kind]
        stack[lv] = (lab, head)
        for k in range(lv + 1, 4):
            stack[k] = None
    path = " > ".join(f"{l} {h}".strip() for l, h in (x for x in stack if x))
    deepest = next((x for x in reversed(stack) if x), ("", ""))
    return deepest[0], deepest[1], path


def _toc_like(line):
    """A line that indexes a section rather than stating one."""
    t = line.strip()
    if not t:
        return False
    if DOTLEAD.search(t):
        return True
    if len(t) > 95:
        return False
    if not SEC_LINE.match(t):
        return False
    # Real provision openers continue into sentence text; index entries do not.
    tail = SEC_LINE.sub("", t).strip(" .:-\u2013")
    return len(tail.split()) <= 9 and not tail.endswith((".", ";", ":"))


def strip_toc(text, run=4):
    """Drop maximal runs of table-of-contents lines, which mimic section structure."""
    lines = text.split("\n")
    flag = [_toc_like(l) for l in lines]
    keep, i, n = [], 0, len(lines)
    while i < n:
        if flag[i]:
            j = i
            while j < n and (flag[j] or not lines[j].strip()):
                j += 1
            if sum(flag[i:j]) >= run:
                i = j
                continue
        if not TOC_HDR.search(lines[i][:50]):
            keep.append(lines[i])
        i += 1
    return "\n".join(keep)


def looks_substantive(t):
    """A provision must read like enacted text, not a heading or TOC fragment."""
    w = t.split()
    if len(w) < 12:
        return False
    if DOTLEAD.search(t):
        return False
    caps = sum(1 for x in w if x.isupper() and len(x) > 2)
    if caps / len(w) > 0.55:
        return False
    return bool(re.search(r"\b(shall|may|must|is|are|will|has|have|means|includes|be|not)\b", t, re.I))


HEADING_LINE = re.compile(r"^[ \t]*([A-Z][A-Za-z'\u2019/&,\- ]{4,70})[ \t]*$")


def split_unnumbered(text, marks):
    """Fallback for codes whose provisions are titled but not numbered."""
    lines = text.split("\n")
    starts = []
    for i, ln in enumerate(lines):
        m = HEADING_LINE.match(ln)
        if not m:
            continue
        h = m.group(1).strip()
        w = h.split()
        if not (2 <= len(w) <= 10) or h.isupper():
            continue
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if not nxt or not nxt[:1].isupper() or len(nxt.split()) < 5:
            continue
        starts.append((i, h))
    out = []
    offs, run = [], 0
    for ln in lines:
        offs.append(run)
        run += len(ln) + 1
    for k, (i, h) in enumerate(starts):
        j = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        body = "\n".join(lines[i + 1:j]).strip()
        if not looks_substantive(body):
            continue
        chap, chap_head, hpath = nearest(marks, offs[i])
        out.append({"section": "", "heading": h, "text": body, "history": None,
                    "chapter": chap, "chapter_heading": chap_head, "heading_path": hpath,
                    "char_offset": offs[i],
                    "unnumbered": True})
    return out


def split_provisions(text):
    text = strip_toc(text)
    marks = context_map(text)
    hits = [(m.start(), m.end(), next(g for g in m.groups() if g)) for m in SEC_RE.finditer(text)]
    out = []
    for i, (s, e, num) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        body = text[e:end].strip()
        if not (40 <= len(body) <= 12000):
            continue
        first_nl = body.find("\n")
        heading = body[:first_nl].strip() if 0 < first_nl < 120 else ""
        if heading and not re.search(r"[.;:]$", heading) and len(heading.split()) <= 14:
            body_txt = body[first_nl + 1:].strip()
        else:
            heading, body_txt = "", body
        if not looks_substantive(body_txt):
            continue
        hist = HIST_RE.search(body_txt)
        chap, chap_head, hpath = nearest(marks, s)
        out.append({"section": num, "heading": heading,
                    "text": HIST_RE.sub("", body_txt).strip(),
                    "history": norm(hist.group(1)).strip() if hist else None,
                    "chapter": chap, "chapter_heading": chap_head, "heading_path": hpath,
                    "char_offset": s})
    return out


def domain_of(*fields):
    blob = " ".join(f or "" for f in fields)
    c, v = len(CRIM_CUE.findall(blob)), len(CIV_CUE.findall(blob))
    if c == 0 and v == 0:
        return None
    return "criminal" if c > v else ("civil" if v > c else None)


def main():
    harvest = json.load(open(ROOT / "data/interim/harvest_all.json"))
    provisions, docmeta = [], []
    for nat in harvest:
        for d in nat["docs"]:
            p = RAW / d["file"]
            if not p.exists():
                continue
            try:
                text, npages, ocr = pdf_text(p) if d["kind"] == "pdf" else html_text(p)
            except Exception:
                continue
            if ocr or len(text) < 1200:
                docmeta.append(dict(d, status="ocr_or_empty", n_provisions=0))
                continue
            yrs = YEAR_RE.findall(text[:4000]) + YEAR_RE.findall(text[-2000:])
            doc_year = max((int(y) for y in yrs), default=None)
            provs = split_provisions(text)
            if len(provs) < 3:
                alt = split_unnumbered(strip_toc(text), context_map(strip_toc(text)))
                if len(alt) > len(provs):
                    provs = alt
            doc_id = d["file"].split(".")[0]
            dom_doc = domain_of(d["anchor"], d["url"], text[:3000])
            seen_txt = set()
            deduped = []
            for pr in provs:
                k = re.sub(r"\W+", "", pr["text"].lower())[:220]
                if k in seen_txt:
                    continue
                seen_txt.add(k)
                deduped.append(pr)
            provs = deduped
            for j, pr in enumerate(provs):
                dom = domain_of(pr.get("heading_path"), pr["heading"]) or dom_doc
                provisions.append({
                    "provision_id": f"{doc_id}-{j:04d}", "doc_id": doc_id,
                    "nation": nat["nation"], "source_url": d["url"],
                    "doc_title": d["anchor"], "doc_kind": d["kind"],
                    "doc_year": doc_year, "ocr": False, "domain": dom, **pr})
            docmeta.append(dict(d, status="ok", n_provisions=len(provs),
                                doc_year=doc_year, n_pages=npages))
    json.dump(docmeta, open(ROOT / "data/interim/docmeta.json", "w"), indent=1)
    with open(ROOT / "data/interim/provisions_raw.jsonl", "w") as f:
        for p in provisions:
            f.write(json.dumps(p) + "\n")
    byn = collections.Counter(p["nation"] for p in provisions)
    byd = collections.Counter(p["domain"] for p in provisions)
    print(f"{len(provisions)} provisions from {len(byn)} Nations / {len(docmeta)} documents")
    print("domain:", dict(byd))
    for n, c in byn.most_common(25):
        print(f"  {c:6d}  {n[:60]}")


if __name__ == "__main__":
    main()
