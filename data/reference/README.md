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

Per-manuscript readings for the decomposition go in `data/reference/readings/<chapter>.tsv`
(one file per reference chapter, e.g. `I.7.tsv`). The first line is a header row with the
columns `unit_id, ms, status, reading, source, note`; `status` is one of `present`,
`absent`, `variant`, `illegible`, `not_collated`. Manuscript ids must exist in
`data/registry/sa_manuscripts.yaml`. Editions are not independent witnesses and are never
counted as manuscripts.
