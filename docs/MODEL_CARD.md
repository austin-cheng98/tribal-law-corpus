# Model card

Following the schema of Mitchell et al. (2019). Two classifiers are described. Only the first
is released as trained weights; the reason is given under Ethical considerations.

---

## 1. Provision-function classifier (released)

### Model details

Linear classifier over character- and word-level tf-idf features (sublinear term frequency,
minimum document frequency 2, word unigrams and bigrams, 200k feature cap) with a linear SVM
(C = 0.5, balanced class weights). A LEGAL-BERT variant fine-tuned on the same folds is
described in the paper and released as code. Trained 2026 by the authors.

### Intended use

Assigning one of eight functional categories (PROCEDURE, SANCTION_REMEDY, GRANT_OF_AUTHORITY,
RIGHT_ENTITLEMENT, PROHIBITION, DEFINITION, SCOPE_APPLICABILITY, OBLIGATION) to a provision of
enacted procedural law, for corpus description and research.

Out of scope: determining anyone's legal rights or obligations; advising on the law of any
Nation; any application where a wrong label affects a person's case. The model does not know
which sovereign's law it is reading and may conflate tribal, federal, and state authority.

### Factors

Performance varies by Nation and by legal domain. The relevant grouping variable is the
source document, not the provision: provisions from one document are not independent of each
other, and evaluation that ignores this overstates performance.

### Metrics

Macro-F1 over the eight classes, with 95% confidence intervals from a bootstrap that resamples
Nations rather than provisions. Significance between training regimes is a sign-flip permutation
test paired over Nations, which is the unit the regimes actually vary over.

### Evaluation data

581 human-verified provisions. Evaluation is on verified labels only; automatic labels are used
for training, never for scoring.

### Training data

The corpus described in `DATASHEET.md`, under three nested regimes matched in size: random,
document-held-out, and leave-one-Nation-out.

### Quantitative analyses

Reported in the accompanying paper across all three regimes and four masking levels, against two
training-free reference points: a rule cascade at 0.386 agreement with human judgment on blind
items and zero-shot entailment at 0.419, both close to the majority-class rate of 0.349 on eight
classes (chance 0.125). Distant supervision recovers little beyond the class prior in this
domain, and the released classifier should be read against that floor.

### Ethical considerations

The eight-class scheme is an analytic convenience imposed by the authors, not a categorization
any Nation uses for its own law. It was not reviewed by any Tribal legal scholar or government.

### Caveats and recommendations

The verified label set comes from a single annotator; intra-annotator kappa is 0.776 and no
inter-annotator figure exists. OBLIGATION has 7 verified instances and its per-class performance
should not be relied on.

---

## 2. Nation-identification probe (code only; weights deliberately not released)

### Model details

Same feature pipeline and linear SVM, trained to predict the authoring Nation from a single
provision. Built as a measurement instrument, not an application.

### Intended use

Measuring how much jurisdiction-identifying surface signal a corpus carries, and how that
measurement changes between evaluation protocols. The generalizable form of this instrument is
the document-identification diagnostic: within a group sharing the label of interest, try to
predict which source document a passage came from. That diagnostic is the contribution and it is
released in full as code, applicable to any corpus.

### Quantitative analyses

On 2,840 provisions from 15 Nations, with training folds matched in size across protocols:
macro-F1 0.697 under a random split, 0.139 with source documents held out, against a permutation
null of 0.065. Four fifths of the apparent signal is document memorization. Redacting Nation
names and normalizing citations changes the random-split figure by 0.018, so surface redaction
does not address the problem.

### Ethical considerations

**Why the weights are withheld.** A trained Nation identifier is, functionally, a jurisdiction
re-identification tool for Indigenous legal data. The paper's own finding is that the honest
re-identification rate at the provision level is low, which is a reason to release derived
provision-level data; it is not a reason to also ship the instrument that attacks it. The code
is released because reproducibility requires it and because other researchers need to run the
diagnostic on their own corpora. Retraining the probe from the released corpus is not difficult,
and withholding weights is therefore a statement of intended use rather than a security control.
It is meant as the former.

### Caveats and recommendations

The document-held-out figure is small but real: 0.139 against a null of 0.065. Do not read it as
evidence that provisions carry no jurisdictional signature, and do not read the random-split
figure as evidence that they carry a strong one.
