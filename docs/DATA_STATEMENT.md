# Data statement

Following the schema of Bender and Friedman (2018).

## A. Curation rationale

The corpus was assembled to support two questions: how much jurisdictional signal a single
provision of enacted law carries, and how that estimate changes under different evaluation
splits. Tribal codes were chosen because the jurisdictions are genuinely independent drafters,
which makes the setting an unusually clean test of cross-jurisdictional generalization.

Inclusion was restricted to criminal law and civil procedure. Provisions were retained only where
the enacted heading structure above them named a legal domain; provisions whose enclosing
structure named neither were excluded rather than assigned by inference.

## B. Language variety

English, United States legal register (BCP-47: `en-US`). The register is enacted statutory
text: numbered sections, deontic modals, defined terms, and internal cross-references. It is
not conversational English and is not representative of any spoken variety.

## C. Speaker demographic

The texts are institutional rather than individual products. Each provision was drafted and
enacted by the legislative body of one of 24 federally recognized Tribal Nations. Individual
drafters are not identified in the source documents and no attempt was made to identify them.
Nations vary widely in population, revenue, and legal-staff capacity; the corpus does not
record these attributes and no inference about them should be drawn from the text.

## D. Annotator demographic

One annotator coded the entire verification sample of 650 queue items. The annotator has
training in law and quantitative social science and is not a citizen of, or affiliated with,
any Nation represented in the corpus. No Tribal legal scholar or Tribal government reviewed
the annotation scheme or the labels. This is the most significant limitation of the resource
and is stated in the paper as such.

Intra-annotator agreement, measured on 50 items that reappeared unmarked later in the queue,
was 0.820 raw and kappa = 0.776. No inter-annotator agreement figure exists.

## E. Speech situation

Enacted law: written, deliberate, edited, and adopted by vote. Publication dates where recorded
in the source documents span the late twentieth century to the present; many documents carry no
reliable date. Text is intended for an audience of tribal courts, counsel, and citizens.

## F. Text characteristics

4,574 provisions: 2,001 criminal law, 2,573 civil procedure. Median length 105 words
(interquartile range 56-196, range 15-600). Provisions are numbered sections, or top-level
subsections where a section was unusually long.

The two sides are not matched on subject matter. The civil side is almost entirely
procedural: 99.8% sits under headings naming pleading, discovery, appellate practice or
rules of procedure. The criminal side is mostly substantive: 64.5% sits under offence and
penal headings against 25.9% under procedural ones, because a Nation publishing a criminal
code usually publishes offences and process in one instrument. Anyone comparing the two
domains should read the contrast as criminal law against civil procedure, not procedure
against procedure.

594 cross-Nation provision pairs exceed a 9-gram Jaccard similarity of 0.80, involving 15.3% of
the corpus. This reuse is concentrated: it affects 25.1% of civil provisions against 2.7% of
criminal ones, and is carried largely by six Oklahoma Nations that share long verbatim passages
of civil procedure. Any model trained on this corpus will see near-duplicate text across
jurisdiction boundaries unless it is screened out.

## G. Recording quality

Text was extracted from PDF and HTML. Documents yielding no extractable character stream
(scanned images without a text layer) were excluded rather than passed through OCR, because OCR
error would have been confounded with Nation. Tables of contents mimic provision structure and
were removed by a line-level detector.

## H. Other

Coverage is biased toward Nations that self-publish their codes. Of 596 Nations indexed by the
National Indian Law Library gateway, 213 link to a code portal that is neither paywalled nor
behind a bot-challenge gate, 87 had retrievable code text, and 24 met the corpus floor
of 60 provisions. The dominant barriers are commercial enclosure and link rot. The
corpus describes the Nations in it and does not generalize to the 574 federally recognized
Nations.

The corpus excludes tribal common law, custom, and oral tradition, which are central to tribal
adjudication and are not reducible to enacted text.
