"""Rule-based labeling functions implementing the Layer-1 priority cascade in annotation/guidelines.md.

Each function returns True if its rule fires on the provision. The cascade is applied in
the documented priority order; the first firing rule assigns the label. These are
deliberately transparent and lexical, which makes the silver labels a conservative
training signal: they encode no Nation-specific information by construction.
"""
import re

FUNCTIONS = ["DEFINITION", "GRANT_OF_AUTHORITY", "SANCTION_REMEDY", "RIGHT_ENTITLEMENT",
             "PROHIBITION", "OBLIGATION", "PROCEDURE", "SCOPE_APPLICABILITY"]

INSTITUTION = r"(?:court|tribunal|judge|justice|council|committee|commission|department|agency|clerk|prosecutor|police|bureau)"

RULES = {
 "DEFINITION": [
   r"^\s*(?:as used|for (?:the )?purposes? of|the following (?:terms|words|definitions))",
   r"[\"“][^\"”]{2,40}[\"”]\s+(?:means|shall mean|is defined)",
   r"\b(?:means|shall mean|is defined as|has the meaning)\b",
 ],
 "GRANT_OF_AUTHORITY": [
   rf"\b{INSTITUTION}\b[^.]{{0,80}}\b(?:shall have|has|is vested with|is granted)\b[^.]{{0,40}}\b(?:jurisdiction|power|authority)",
   r"\bis hereby (?:established|created|ordained|constituted|vested|authorized)\b",
   r"\bthere (?:is|are) hereby (?:established|created)\b",
   rf"\b{INSTITUTION}\b[^.]{{0,60}}\bis authorized to\b",
   r"\bshall (?:have|exercise) (?:exclusive |original |concurrent |appellate )*jurisdiction\b",
 ],
 "SANCTION_REMEDY": [
   r"\b(?:shall be )?(?:punish(?:ed|able)|sentenc(?:ed|e)|fined|imprison(?:ed|ment))\b",
   r"\bfine (?:of )?not (?:more|less) than\b|\bpenalty of\b|\bsubject to a fine\b",
   r"\b(?:contempt of court|forfeiture|revocation of|shall be revoked|shall be dismissed)\b",
   r"\b(?:damages|injunctive relief|restitution) (?:shall|may) be (?:awarded|granted|imposed)\b",
 ],
 "RIGHT_ENTITLEMENT": [
   r"\b(?:shall be|is|are) entitled to\b",
   r"\b(?:shall have|has) the right to\b",
   r"\bright to (?:counsel|a jury|be heard|confront|remain silent|appeal)\b",
   r"\bshall not be (?:denied|deprived of|compelled)\b",
 ],
 "PROHIBITION": [
   r"\bno (?:person|party|one|member|officer|defendant)\b[^.]{0,60}\bshall\b",
   r"\b(?:shall|may) not\b",
   r"\b(?:it shall be|is) unlawful\b|\bis prohibited\b|\bshall be prohibited\b",
 ],
 "OBLIGATION": [
   r"\bit shall be the duty of\b",
   r"\b(?:is|are) required to\b",
   r"\bmust\b",
 ],
 "PROCEDURE": [
   r"\bwithin\s+(?:\w+\s+)?\(?\d+\)?\s+(?:calendar |business |working )?days?\b",
   r"\bshall (?:be )?(?:file[d]?|serve[d]?|notif(?:y|ied)|schedul(?:e|ed)|set for hearing|docket)\b",
   r"\b(?:summons|complaint|pleading|motion|petition|subpoena|notice of appeal)\b[^.]{0,60}\bshall\b",
   r"\bupon (?:motion|application|petition|request) of\b",
   r"\bhearing shall be (?:held|conducted|set)\b",
 ],
 "SCOPE_APPLICABILITY": [
   r"\bthis (?:title|chapter|code|act|section|ordinance)\b[^.]{0,50}\b(?:shall )?appl(?:y|ies)\b",
   r"\bshall (?:not )?apply to\b",
   r"\b(?:effective date|shall take effect|severability|savings clause|repealed)\b",
   r"\bnothing (?:in this|herein) (?:title|chapter|code|section)\b",
 ],
}
COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in RULES.items()}

DEF_HEADING = re.compile(r"^\s*definitions?\b", re.I)
SCOPE_HEADING = re.compile(r"\b(?:purpose|scope|applicability|short title|effective date|severability|construction)\b", re.I)


def fires(label, text, heading=""):
    if label == "DEFINITION" and DEF_HEADING.search(heading or ""):
        return True
    if label == "SCOPE_APPLICABILITY" and SCOPE_HEADING.search(heading or ""):
        return True
    return any(r.search(text) for r in COMPILED[label])


def label_provision(text, heading=""):
    """Apply the cascade. Returns (label, n_rules_fired) or (None, 0) when nothing fires."""
    hits = [f for f in FUNCTIONS if fires(f, text, heading)]
    return (hits[0], len(hits)) if hits else (None, 0)


# ---- Layer 2: deontic modality (multi-label) ----
MODALITY = {
 "MUST_NOT": re.compile(r"\b(?:shall not|may not|must not|no \w+ shall|is prohibited|unlawful)\b", re.I),
 "MAY":      re.compile(r"\b(?:may|is authorized to|in the discretion of|is permitted|at the option of)\b", re.I),
 "MUST":     re.compile(r"\b(?:shall|must|is required to|it shall be the duty)\b", re.I),
}
DEF_FRAME = re.compile(r"\bshall (?:mean|be (?:construed|deemed|interpreted))\b", re.I)


NEG_SPAN = re.compile(r"\b(?:shall not|may not|must not|no \w+(?:\s+\w+)? shall)\b", re.I)


def modality(text):
    """Multi-label deontic modality. Negative operators mask the 'shall'/'may' they contain."""
    t = DEF_FRAME.sub(" ", text)
    masked = NEG_SPAN.sub(" \u00a7NEG\u00a7 ", t)          # remove negated duties before scanning affirmatives
    out = []
    if MODALITY["MUST_NOT"].search(t):
        out.append("MUST_NOT")
    if MODALITY["MAY"].search(masked):
        out.append("MAY")
    if MODALITY["MUST"].search(masked):
        out.append("MUST")
    return out or ["NONE"]


# ---- Layer 4: cross-reference type ----
XREF = {
 "FEDERAL": re.compile(r"\b\d+\s*U\.?\s?S\.?\s?C\.?|\bC\.?F\.?R\.?\b|Indian Civil Rights Act|"
                       r"\bPublic Law\s+\d|\bfederal (?:law|statute|court|government)\b|"
                       r"\bUnited States (?:Code|Attorney|District Court)\b|25 U\.S\.C", re.I),
 "STATE":   re.compile(r"\bstate (?:law|statute|court|of)\b|\bState of [A-Z]\w+", re.I),
 "CUSTOM":  re.compile(r"\b(?:tribal )?(?:custom|customs|customary law|tradition(?:al|s)?|"
                       r"unwritten law|elders?|cultural (?:values|practices))\b", re.I),
 "INTERNAL": re.compile(r"\b(?:section|chapter|title|article)\s+\d+[-.\d]*\b|\bthis (?:code|title|chapter)\b", re.I),
}


def xrefs(text):
    return [k for k, r in XREF.items() if r.search(text)]
