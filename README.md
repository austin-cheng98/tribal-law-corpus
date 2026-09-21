# Tribal criminal law and civil procedure: corpus and split-ladder experiments

Code and derived data for a corpus of 4,574 provisions of criminal law and civil procedure from
the enacted codes of 24 federally recognized Tribal Nations, and for the experiments measuring
how much of the apparent cross-jurisdictional signal in such a corpus is source-document
memorization.

## The finding, in one paragraph

Multi-jurisdictional corpora arrive as source documents, one or a few per jurisdiction. The
jurisdiction label is then constant within a document, documents are individually recognizable
from surface form, and a random split over passages puts the same document on both sides of the
partition. A model can score well by recognizing the document and reading the label off it. On
this corpus a Nation-identification probe reaches 0.697 macro-F1 under a random split and 0.139
with source documents held out, against a permutation null of 0.065.

Before trusting a random split, run the diagnostic: within each group sharing the label of
interest, train a classifier to identify which source document a passage came from. Here it
reaches 0.858 macro-F1. It costs one linear model and needs no held-out data.

```
python src/exp_doc_probe.py --corpus data/processed/corpus.jsonl
```

## Layout

```
src/01..11_*.py     acquisition through gold-label ingestion, in order
src/exp_*.py        experiments; each writes one JSON file to results/
src/check_*.py      robustness checks on those experiments
src/data.py         corpus loading, stratified sampling, the global seed
src/masking.py      the M0-M3 ablation ladder
src/labeling_functions.py   the rule cascade
src/encoder.py      fine-tuned encoder classifier
src/nli_annotate.py zero-shot entailment annotator
src/build_workbook.py       builds the annotation workbook from the sample
data/processed/     the released corpus
docs/               data statement, datasheet, model card, removal procedure,
                    annotation guidelines
results/            experiment outputs
```

Raw harvested documents are not distributed; see Governance. `src/01..07` regenerate the corpus
from source, and every record carries the URL it came from.

## Reproducing

```
pip install -r requirements.txt
python src/exp_doc_probe.py        # documents are recognizable
python src/exp_probe_matched.py    # Nation identification, both protocols, size-matched
python src/exp_probe_domain.py     # the same, within one legal domain
python src/exp_diffusion.py        # cross-Nation near-duplicate detection
python src/exp_comparative.py      # divergence between Nations, by domain
python src/exp_classify.py         # provision function across the split ladder
python src/exp_source_diversity.py # what predicts the distortion, over four labels
python src/exp_template_adoption.py # borrowed drafting on each side of the corpus
python src/check_probe_cluster.py  # the probe interval at document and Nation level
```

Every experiment is seeded from `SEED` in `src/data.py`. Confidence intervals resample Nations
rather than provisions, because provisions from one Nation are not independent. Significance
between regimes is a sign-flip permutation test paired over Nations.

To rebuild the corpus from source, run `src/01_index_sources.py` through
`src/07_build_corpus.py` in order, then `08_domain.py`, `09_silver.py`. The harvester respects
`robots.txt` and rate-limits per host. It does not attempt to pass bot-challenge gates, and
several Nations are absent from the corpus for that reason.

## Splits

Do not evaluate this corpus with a random split over provisions. Three regimes are implemented
in `src/exp_classify.py`:

- `R1_random` - training drawn from the whole corpus minus the test set
- `R1d_doc_heldout` - training excludes the test documents, keeps the test Nation
- `R2_lono` - training excludes the test Nation entirely

All three are subsampled to the size of the smallest, so a difference between them reflects
composition rather than training-set size. 15.3% of provisions have a near-verbatim counterpart
in another Nation, so screen near-duplicates before holding a Nation out or the held-out
Nation's text leaks into training through its neighbours.

## Labels

Domain (criminal / civil) comes from the enacted heading structure, never from provision
wording, so it is not circular with respect to lexical models trained to predict it. The two
sides are not matched on subject matter: the civil side is almost entirely procedural, while
the criminal side is mostly substantive, because a Nation publishing a criminal code usually
publishes offences and process in one instrument. Read the contrast as criminal law against
civil procedure.

Provision function has eight classes and two automatic sources, a rule cascade and a zero-shot
entailment model. Both sit close to the majority-class rate when checked against human
judgment: 0.386 and 0.419 against a majority class of 0.349. 581 provisions carry a
human-verified label; evaluate on those, and treat the automatic labels as training signal
only.

## Governance

The corpus is derived from the public, enacted law of sovereign Nations. Public availability is
not redistribution consent.

Released as derived data - provision spans with structural metadata, labels, and source URLs -
rather than as mirrored source documents. No Tribal government or Tribal legal scholar was
consulted in building it, so the CARE principle of Authority to control is not satisfied by this
release. `docs/REMOVAL_REQUESTS.md` documents how a Nation may have its material removed;
removal takes precedence over reproducibility.

Trained Nation-identifier weights are deliberately not distributed. See `docs/MODEL_CARD.md`.

Source URLs ship with each record. Provenance and independent verification need them, and each
points to a document the Nation itself published, but they also lower the cost of mass
re-collection and the removal procedure does not reach copies already distributed.

The corpus covers English-language enacted codes. It excludes tribal common law, custom, and
oral tradition, which are central to tribal adjudication and are not reducible to enacted text.
It describes the Nations in it and does not generalize to the 574 federally recognized Nations.

Do not use any of this to determine anyone's legal rights or obligations.

## License

Enacted law is not ours to license. The code in `src/` and the annotations in
`data/processed/` are released for research use.
