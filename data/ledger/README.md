# Test-evaluation ledger

`test_evaluations.jsonl` records every time the TEST gold (`data/annotations/gold/<witness>/test.csv`)
was scored. It is append-only: lines are added by `hevajra_matrix.evaluation.gate.ledger_append`
and never edited or removed, and it is committed with the repository. Scoring dev gold is not
recorded here.

The ledger makes the confirmatory claim checkable. A report is confirmatory only if the
preregistration is frozen, the instrument digest is the preregistered one, every ledger line of
that digest carries the current preregistration sha256, and the test set was scored exactly once
for that digest. Otherwise the report reads "exploratory; test set scored N times by K instrument
versions" (`evaluation.gate.evaluate`, `gate.ledger_summary`). It also replaces a drift canary:
runs of different instrument versions can be compared line by line.

## Line format

One JSON object per line with exactly these keys (`gate.ledger_record` builds it):

| Key | Content |
|---|---|
| `ts` | UTC time of the scoring, ISO 8601 |
| `instrument_digest` | `InstrumentId.digest()` of the collator that was scored |
| `prereg_sha256` | sha256 of `config/preregistration.yaml` at scoring time |
| `git_commit` | commit of the code, or null outside a git checkout |
| `metrics` | per source (e.g. `claude:consensus`, `dp:zero`, `dp:anchor`), the metrics of `AlignmentScores.metrics()` |
| `gate` | `pass` or `fail`: the outcome of gate G1 |

Example (wrapped here for reading; one line in the file):

```
{"ts": "2026-11-02T10:15:00+00:00", "instrument_digest": "3f2a...", "prereg_sha256": "9c1b...",
 "git_commit": "ffc00c6...", "metrics": {"claude:consensus": {"link_f1": 0.86}}, "gate": "pass"}
```

Order of use: `check_g1` -> `ledger_append(ledger_record(...))` -> `evaluate` with the
`ledger_summary` read after the append.
