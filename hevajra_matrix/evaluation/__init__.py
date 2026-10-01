"""Evaluation: gold scoring against controls, sentinel facts, gates and the test ledger (synthesis 5).

    gold       committed gold format (``load``/``save``, verdict conversion), ``score`` of any
               ``Alignment`` against gold, the metrics, bootstrap intervals and McNemar
    windows    dev regions from the preregistration and the probability sample of test
               windows (re-exported by ``gold``)
    resample   kappa, positive agreement, window-cluster and paired bootstraps, McNemar
               (re-exported by ``gold``)
    sentinels  the 8 sentinel check kinds at stages ingest, proposal and final
    gate       G0-G4, report level, confirmatory flag, the append-only ledger

Nothing here calls a model: ``evaluate --gold dev --baselines-only`` scores the DP baselines
and any imported external alignment without an API key. Submodules are imported explicitly.
"""
