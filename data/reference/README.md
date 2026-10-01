# data/reference/ (not committed)

Sanskrit critical editions are under copyright, so the repository keeps only unit
coordinates. Put your own electronic texts here; the file stem is the witness `id`
from `data/registry/witnesses.yaml`:

```
data/reference/sa_snellgrove1959.tsv      default row identifiers (reference scheme)
data/reference/sa_tripathi_negi2001.tsv
data/reference/sa_conlon2022.tsv
data/reference/sa_farrow_menon1992.tsv
```

Format: tab-separated, lines starting with `#` are comments.

```
I.1.p01<TAB>evam maya srutam ekasmin samaye bhagavan ...
I.1.1<TAB>vajrasattvo bhavet kasmat ...
I.2.5<TAB><TAB>LACUNA        the edition reports a physical gap in its manuscripts
I.2.6<TAB><TAB>ABSENT        the edition has no such verse (another edition has it)
```

## The Sanskrit reference

When the file named by `config/run.yaml: witnesses.sanskrit_reference` (default
`sa_snellgrove1959`) is here, `ingest` makes its units the reference of the run: the matrix
rows become the Sanskrit units (Snellgrove ids), and both the Derge and T0892 are aligned
witnesses (Sanskrit -> Tibetan and Sanskrit -> Chinese, each with its own collation, matrix,
gates and report; the Derge's outputs go to `<run>/witnesses/bo_derge_D417_418/`). Units
flagged `LACUNA` or `ABSENT` have no text and are no matrix rows. The scope lines of every
report then say "relative to the Sanskrit reference ...". Before collating, write the
collator examples of the two new language pairs (`data/codebook/collate_examples.sa-bo.yaml`,
`collate_examples.sa-zh.yaml`; see `docs/llm-tasks.md`). Gold, verdicts and topic labels made
against the Sanskrit units live under `data/annotations/by_reference/<reference>/`.

Without the file the provisional Derge stays the reference and nothing changes.

## Per-manuscript readings (E3)

Per-manuscript readings for the decomposition go in `data/reference/readings/<chapter>.tsv`
(one file per reference chapter, e.g. `I.7.tsv`). The first line is a header row with the
columns `unit_id, ms, status, reading, source, note`; `status` is one of `present`,
`absent`, `variant`, `illegible`, `not_collated`. `unit_id` is a reference unit id: a
Snellgrove id, or a Derge segment id (`D417:8a.6.1`) while the Derge is the reference.

Manuscript ids must exist in `data/registry/sa_manuscripts.yaml`. Editions are not
independent witnesses and are never counted as manuscripts: a row whose `ms` is an
edition's witness id (e.g. `sa_tripathi_negi2001`) is set aside and listed as
`editions_ignored`; any other unknown id stops `stats` with an error.

`stats` uses the readings for E3: a deviating unit is `vorlage_ms` when an informative
manuscript lacks or varies it, `insufficient` below `gates.g4.min_manuscripts_per_unit`
informative manuscripts, and otherwise `shared` (the co-witness deviates too) or `residual`.
E3 is estimable only under the Sanskrit reference (the co-witness of the Chinese is then the
Derge, and vice versa); under the Derge reference it prints `NOT_ESTIMABLE: G4: reference is
the co-witness` and the decomposition is written to `stats/details.json` as description only.
