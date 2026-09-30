# Topic labels: file format

One CSV file per reference text, named after the reference witness id in
`data/registry/witnesses.yaml`, e.g. `bo_derge_D417_418.csv`. It holds the human topic
label of every reference unit. The labels are the exposure of the sensitive-vs-neutral
contrast (E4) and define the arms of the over-attribution experiment. The file is read
by `hevajra_matrix.topics.load_labels` and written by `write_labels`.

## Columns

Exactly these nine columns, in any order (`write_labels` writes this one), UTF-8 (a
byte-order mark is tolerated):

| Column | Content |
|---|---|
| `unit_id` | reference unit id, e.g. `D418:17b.6.2`; unique in the file |
| `fingerprint` | the unit's text fingerprint when it was labelled (`core.textnorm.fingerprint`) |
| `topics` | the first coder's confirmed topics; empty while the unit is not labelled yet |
| `prelabel_topics` | the verified Claude pre-label (task T3) that the first coder saw; empty when there was no usable pre-label |
| `coder` | the first coder's name or initials (required when `topics` is filled) |
| `date` | date of the first coder's label, `YYYY-MM-DD` |
| `second_topics` | the second coder's topics; empty when the unit is not double-coded |
| `second_coder` | the second coder (required when `second_topics` is filled; must differ from `coder`) |
| `second_date` | date of the second label, `YYYY-MM-DD` |

A file with any other column is refused. In particular, a unit's text never goes into
this file (licence rule: committed annotations hold ids and labels only).

## Topic values

Topic names come from `data/codebook/topics.yaml`: `sexual`, `female_agent`,
`flesh_food`, `harm`, `theft`, `impure_substance`, `bone_corpse`, `ritual` (the
sensitive group, Tokushige 2026's eight categories), `neutral`, `frame` and
`mantra_control`. Several topics in one cell are separated by `;`, e.g.
`sexual;female_agent`. On loading:

- an unknown topic is an error that names the line and lists the allowed values;
- `neutral` must appear alone;
- every invalid row is reported at once, and nothing is loaded until all are fixed.

A unit's group (`topic_group`) is the first of `sensitive`, `mantra_control`, `frame`,
`neutral` that holds one of its topics, or `unlabelled` when it has none.

## Rules of the workflow

- **Blind to the Chinese.** Coders label the reference text only and never look at the
  Chinese translation. The pre-labeller never sees it either.
- **The second coder is blind to the pre-labels** (critique A6): the second coder's sheet
  has no pre-label columns. Topic kappa (`human_agreement`, gate G4) compares `topics`
  with `second_topics` only. Agreement of the pre-labels with the humans
  (`prelabel_agreement`) is reported separately and never enters topic kappa.
- **Only measurement-grade pre-labels are committed.** `prelabel_topics` holds
  `Prelabel.committed_topics`: the verified topics of an answer from the requested model.
  A refused, truncated, invalid or skipped batch leaves it empty, and so does an answer
  served by a fallback model. That answer may be shown on the sheet as a hint, but it is
  never committed.
- **Stale labels.** A label applies only while its `fingerprint` matches the unit's
  current text. `current_labels` lists the stale ones for relabelling; they are never
  used silently.
