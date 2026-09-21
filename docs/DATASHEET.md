# Datasheet

Following the schema of Gebru et al. (2021). Answers are specific to this release; questions
that do not apply are marked as such rather than omitted.

## Motivation

**For what purpose was the dataset created?** To measure how much jurisdictional signal a single
provision of enacted law carries, and how that measurement depends on the evaluation split. No
corpus of Tribal law existed for NLP research, and the setting is an unusually clean instance of
a general problem: labels nested inside recognizable source documents.

**Who funded the creation of the dataset?** No external funding.

## Composition

**What do the instances represent?** Individual provisions of enacted Tribal law: a numbered
section, or a top-level subsection where a section was unusually long.

**How many instances are there?** 4,574 provisions from 24 federally recognized Tribal Nations,
extracted from 89 documents served by 24 distinct Nation-operated domains. 2,001 are criminal
law and 2,573 civil procedure. The two sides are not matched on subject matter. The civil side is almost entirely
procedural: 99.8% sits under headings naming pleading, discovery, appellate practice or
rules of procedure. The criminal side is mostly substantive: 64.5% sits under offence and
penal headings against 25.9% under procedural ones, because a Nation publishing a criminal
code usually publishes offences and process in one instrument. Anyone comparing the two
domains should read the contrast as criminal law against civil procedure, not procedure
against procedure.

**Does the dataset contain all possible instances?** No. It is a sample constrained by
accessibility, described under Collection process. It is not a random sample and is biased
toward Nations that self-publish their codes.

**What data does each instance consist of?** Provision text, the full enacted heading stack
above it (title, chapter, article, part), section and chapter identifiers, document identifier
and title, Nation, serving platform, legal domain and the basis on which it was assigned,
character offset in the source document, word count, source URL, detected deontic modality,
internal cross-references, a near-duplicate flag, and where present the automatic and verified
function labels.

**Is there a label or target?** Two automatic label sources (a rule cascade and zero-shot
entailment) cover the corpus. 581 provisions carry a human-verified function label over eight
classes. Class counts: PROCEDURE 203, SANCTION_REMEDY 165, GRANT_OF_AUTHORITY 69,
RIGHT_ENTITLEMENT 46, PROHIBITION 42, DEFINITION 29, SCOPE_APPLICABILITY 20, OBLIGATION 7.

**Is any information missing?** Enactment and amendment dates are absent or unreliable for most
documents. Where a document carried no usable date, none was inferred.

**Are relationships between instances made explicit?** Yes. Provisions carry their document and
Nation identifiers, and cross-Nation near-duplicate pairs are flagged.

**Are there recommended data splits?** Yes, and this is the central methodological point of the
accompanying paper. A random split over provisions puts text from the same source document on
both sides of the partition and does not measure cross-jurisdictional generalization. Use the
document-held-out and leave-one-Nation-out splits, and report both.

**Are there errors or redundancies?** 15.3% of provisions have a near-verbatim counterpart in
another Nation. A manual audit of 24 sampled provisions found the structural domain label
consistent with content in all but two cases, both involving cross-referencing between titles.

**Does the dataset contain confidential or offensive content?** It is public, enacted law. Some
criminal-procedure provisions describe offences, including sexual offences and offences against
children, in the clinical register of statutory drafting. No personal data about identifiable
individuals is present; the texts are general legislation rather than case records.

## Collection process

**How was the data acquired?** From the National Indian Law Library's Tribal law gateway, which
indexes code pages for 596 Nations. 213 link to a portal that is neither behind a commercial
paywall nor behind a bot-challenge gate. Bot-challenge gates were not circumvented. Probing
those portals left 87 Nations with retrievable code text, and a breadth-first harvester
respecting `robots.txt` and a per-host rate limit retrieved 89 documents that survived parsing,
from which 24 Nations met the corpus floor of 60 provisions.

**Who was involved?** The authors only. No crowdworkers or contractors were used and no one was
paid for annotation.

**Over what timeframe was the data collected?** A single harvest, 2026. Each record carries the
URL it was retrieved from.

**Were the individuals notified or did they consent?** No. The instances are enacted statutes,
not data about individuals, so individual consent is not the applicable question. The applicable
question is Nation authority over Nation-produced data, and it is answered in Ethics below: it
was not obtained.

**Has an ethical review process been conducted?** No institutional review was sought; the
material is public law and involves no human subjects. No Tribal review was obtained, which is
the limitation that matters here.

## Preprocessing, cleaning, labeling

Text was extracted from PDF with page-level boilerplate removal and from HTML with content
isolation. Table-of-contents lines were removed by a line-level detector, and a candidate was
retained only if it read as enacted text rather than an index entry. Scanned documents without a
text layer were excluded rather than OCR'd.

Domain labels derive from the enacted heading structure, never from provision wording, so they
cannot be circular with respect to lexical models trained to predict them. Function labels come
from a rule cascade and a DeBERTaV3 zero-shot entailment model; a stratified sample was verified
by hand, two thirds of it blind to the automatic suggestion.

**Is the raw data saved?** Yes, alongside the processed corpus. The harvester, extractor, and
labeling code are released.

## Uses

**What tasks has the dataset been used for?** Nation identification, provision-function
classification, cross-Nation near-duplicate detection, and comparison of provision-function and
deontic-modality distributions between Nations.

**What should the dataset not be used for?** It must not be used to determine anyone's legal
rights or obligations, to rank Nations or score legal systems, or to train systems that present
generated text as the law of a Nation. Models trained on it may confuse tribal, federal, and
state authority. Textual similarity between two codes is not evidence of who copied whom.

**Is there anything that might cause unfair treatment?** The corpus over-represents Nations with
the administrative capacity to self-publish. Treating it as representative of Tribal law
generally would systematically misdescribe Nations absent from it.

## Distribution

**How is it distributed?** As derived data: provision text spans with structural metadata,
labels, and source URLs, rather than as mirrored source documents.

**Under what license?** Enacted law is not the authors' to license. The derived annotations and
code are released for research use. A Nation may request removal of its material by the
procedure documented in the repository, and such requests will be honored without requiring a
justification.

**Are there export controls or regulatory restrictions?** None known.

## Maintenance

**Who maintains it?** The authors. Removal requests take precedence over reproducibility: if a
Nation asks for its material to be withdrawn, it will be removed from the distributed corpus and
the change recorded, and the affected results will be reported as no longer reproducible from
the public release rather than the request being refused on reproducibility grounds.

**Will it be updated?** Corrections to extraction and labeling errors, yes. Expansion to
additional Nations is not planned without a governance arrangement that the current release
lacks.
