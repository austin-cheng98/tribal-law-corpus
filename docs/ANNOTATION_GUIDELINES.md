# Annotation Guidelines: Tribal Procedural Law Provisions

Version 1.2. Unit of annotation: one **provision** (a numbered section or, where a section
exceeds 12,000 characters, its top-level subsection).

Annotators label four layers. Layers 1 and 2 are required for every provision; layers 3
and 4 are span-level and are annotated only on the gold subset.

---

## Layer 1 — Provision function (single label, 8 classes)

Assign exactly one label using the **priority cascade** below. Where a provision performs
several functions, the *earliest* matching rule wins. The cascade exists to make the label
reproducible; it is not a claim that provisions have only one function.

| # | Label | Applies when the provision... | Test |
|---|---|---|---|
| 1 | `DEFINITION` | stipulates the meaning of a term for the instrument | contains "means", "shall mean", "is defined as", "includes" in a definitional frame |
| 2 | `GRANT_OF_AUTHORITY` | confers jurisdiction, power, or competence on an **institution** | "the Court shall have jurisdiction", "the Council is authorized", "is hereby established" |
| 3 | `SANCTION_REMEDY` | states a penalty, remedy, or enforcement consequence | fines, imprisonment, contempt, dismissal, injunction, revocation |
| 4 | `RIGHT_ENTITLEMENT` | confers a right, privilege, or immunity on a **person** | "is entitled to", "has the right to", "may not be denied" |
| 5 | `PROHIBITION` | forbids conduct | "shall not", "may not", "no person shall", "it is unlawful" |
| 6 | `OBLIGATION` | imposes a mandatory duty to act | "shall file", "must serve", "is required to" |
| 7 | `PROCEDURE` | specifies the sequence, timing, or form of a process step | deadlines, filing mechanics, order of proceedings, service rules |
| 8 | `SCOPE_APPLICABILITY` | delimits when, where, or to whom the instrument applies | applicability, effective dates, severability, savings clauses |

**Cascade notes.**
- A provision that *both* defines an offence and sets its penalty is `SANCTION_REMEDY` only
  if the penalty is the operative content; if it states the forbidden conduct and then
  cross-references a penalty elsewhere, label `PROHIBITION`.
- Jurisdictional statements about a **court** are `GRANT_OF_AUTHORITY`, not `SCOPE_APPLICABILITY`.
- Timing rules addressed to a party ("the defendant shall answer within 20 days") are
  `PROCEDURE`, not `OBLIGATION`: `OBLIGATION` is reserved for duties whose content is not a
  step in an adjudicative sequence.

## Layer 2 — Deontic modality (multi-label, 4 classes)

Mark every modality the provision expresses on its principal clause.

- `MUST` — shall, must, is required to, it is the duty of
- `MAY` — may, is authorized to, in the discretion of, is permitted
- `MUST_NOT` — shall not, may not, no ... shall, is prohibited
- `NONE` — purely constitutive or declarative text with no deontic operator

"Shall" in a definitional frame ("'Court' shall mean") is **not** `MUST`; it is `NONE`.

## Layer 3 — Institutional actor spans (10 types)

Tag the minimal noun phrase denoting the actor.

`COURT` · `JUDGE` · `CLERK` (clerk, court administrator, bailiff) · `PROSECUTOR` ·
`DEFENSE` (defender, advocate, counsel for the accused) · `LAW_ENFORCEMENT` (police,
officer, game warden) · `COUNCIL` (legislature, business committee, general council) ·
`PARTY` (plaintiff, defendant, petitioner, respondent, litigant) · `CITIZEN_MEMBER`
(enrolled member, Indian, person subject to jurisdiction) · `EXTERNAL_SOVEREIGN`
(United States, federal agency, State, county).

Tag each mention, including pronominal repetitions within the provision.

## Layer 4 — Relational and referential annotation

**Condition / exception spans.** Mark `CONDITION` (if, when, unless, upon, provided that,
subject to) and `EXCEPTION` (except, notwithstanding, other than) over the full
subordinate clause.

**Cross-reference type.** For each citation in the provision, mark one of
`INTERNAL` (this Nation's own code), `FEDERAL` (U.S.C., C.F.R., ICRA, Public Law),
`STATE` (state statutes or courts), `CUSTOM` (tribal custom, tradition, customary law,
unwritten law, elders).

**Relations** (gold subset only): `ACTOR–has_power_over–TARGET`,
`ACTOR–owes_duty_to–TARGET`, `RULE–has_exception–CONDITION`,
`COURT–has_jurisdiction_over–PARTY`.

---

## Adjudication

Disagreements are resolved by re-reading the cascade in order and applying the first rule
that fires. Where the cascade does not resolve the disagreement, the provision is marked
`CONTESTED` and excluded from the gold set; the contested rate is reported.

## What annotators must not do

Do not infer the quality, fairness, or adequacy of a provision. Do not compare Nations.
Do not label based on the identity of the Nation. If a provision cannot be understood
without external context, mark `INSUFFICIENT_CONTEXT` rather than guessing.
