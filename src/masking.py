"""Lexical-cue ablation ladder.

M0  raw text
M1  + name redaction        capitalised non-sentence-initial tokens outside a generic legal stoplist
M2  + numeric normalisation  section numbers, citations, amounts, day counts -> placeholders
M3  + statistical ablation   top-k Nation-discriminative terms by Monroe et al. (2008) log-odds

M3's term list must be fitted on training folds only; `fit_discriminative_terms` takes an
explicit set of provisions so that test text never informs the ablation.
"""
import collections, math, re

GENERIC = {
 # institutional and structural vocabulary shared across jurisdictions
 "Court","Courts","Tribal","Tribe","Nation","Council","Judge","Judges","Justice","Chief",
 "Clerk","Chapter","Title","Section","Sections","Article","Part","Code","Act","Rule","Rules",
 "Reservation","Constitution","Bylaws","Ordinance","Committee","Commission","Department",
 "Trial","Appellate","Appeals","Supreme","District","Magistrate","Peacemaker","Prosecutor",
 "Defendant","Plaintiff","Petitioner","Respondent","Complainant","Party","Parties","Juvenile",
 "President","Chairman","Chairperson","Secretary","Treasurer","Governor","Board","Office",
 "The","This","That","These","Those","A","An","If","When","Where","No","Any","All","Each",
 "Upon","Within","Notwithstanding","Provided","Except","Whenever","In","On","At","For","To",
 "United","States","Federal","State","Indian","Bureau","Congress","Public","Law","Attorney",
 "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday",
 "January","February","March","April","May","June","July","August","September","October",
 "November","December",
}
CAP = re.compile(r"\b([A-Z][A-Za-z'’\-]{1,})\b")
SENT_START = re.compile(r"(?:^|[.;:!?]\s+|\n\s*|\(\w\)\s+|\d+\.\s+)$")
NUM_PATTERNS = [
 (re.compile(r"\b\d+\s*U\.?\s?S\.?\s?C\.?(?:\s*§+\s*[\d\-.()a-z]+)?", re.I), " §CITE "),
 (re.compile(r"\b\d+\s*C\.?\s?F\.?\s?R\.?(?:\s*§+\s*[\d\-.()a-z]+)?", re.I), " §CITE "),
 (re.compile(r"§+\s*[\d]+[\d\-.]*"), " §SEC "),
 (re.compile(r"\b(?:Section|Chapter|Title|Article|Rule)\s+[\d]+[\d\-.]*", re.I), " §SEC "),
 (re.compile(r"\$\s?[\d,]+(?:\.\d\d)?"), " §AMT "),
 (re.compile(r"\b\d[\d,]*\b"), " §NUM "),
]
TOKEN = re.compile(r"[a-z][a-z\-']+")


def redact_names(text):
    out, last = [], 0
    for m in CAP.finditer(text):
        w = m.group(1)
        if w in GENERIC:
            continue
        if SENT_START.search(text[max(0, m.start() - 3):m.start()]):
            continue
        out.append(text[last:m.start()] + "§NAME")
        last = m.end()
    return "".join(out) + text[last:]


def normalise_numbers(text):
    for pat, rep in NUM_PATTERNS:
        text = pat.sub(rep, text)
    return text


def fit_discriminative_terms(provisions, top_k=40, min_count=5):
    """Monroe et al. (2008) log-odds with an informative Dirichlet prior.

    Returns {nation: set(term)} of the top_k most Nation-distinctive terms.
    `provisions` must come from training folds only.
    """
    by_nat = collections.defaultdict(collections.Counter)
    for p in provisions:
        by_nat[p["nation"]].update(TOKEN.findall(p["text"].lower()))
    total = collections.Counter()
    for c in by_nat.values():
        total.update(c)
    n_tot = sum(total.values())
    a0 = 500.0                                     # prior strength
    out = {}
    for nat, cnt in by_nat.items():
        n_i = sum(cnt.values())
        rest = {w: total[w] - cnt[w] for w in total}
        n_j = n_tot - n_i
        scores = []
        for w, c_iw in cnt.items():
            if total[w] < min_count:
                continue
            a_w = a0 * total[w] / n_tot
            l_i = math.log((c_iw + a_w) / (n_i + a0 - c_iw - a_w))
            l_j = math.log((rest[w] + a_w) / (n_j + a0 - rest[w] - a_w))
            var = 1.0 / (c_iw + a_w) + 1.0 / (rest[w] + a_w)
            scores.append(((l_i - l_j) / math.sqrt(var), w))
        scores.sort(reverse=True)
        out[nat] = {w for _, w in scores[:top_k]}
    return out


def ablate_terms(text, terms):
    if not terms:
        return text
    pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, terms), key=len, reverse=True)) + r")\b", re.I)
    return pat.sub("§CUE", text)


def apply(text, level, terms=None):
    if level >= 1:
        text = redact_names(text)
    if level >= 2:
        text = normalise_numbers(text)
    if level >= 3:
        text = ablate_terms(text, terms or set())
    return re.sub(r"\s+", " ", text).strip()
