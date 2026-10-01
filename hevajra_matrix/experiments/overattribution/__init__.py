"""Over-attribution experiment v2 (synthesis section 8; critique A3, B12).

Does Claude still assert a translator motive for an omission in the Chinese translation
when the evidence shown explains it by the source side, and more so for sensitive content?

    design     items (coordinates only), evidence lines, balanced and seeded trial schedule
    run        T4 subject and T5 scorer requests (fallback off), the run loop
    score      schemas, answer verification, outcomes
    lexical    negation-aware lexical motive baseline (comparison only)
    analysis   item-level H1/H2 (Holm), exploratory H3, Manski refusal bounds, two-phase correction
    human      human sample, blind coding sheet (opaque ids), human codes, scorer kappa
"""

from .analysis import AnalysisParams, ExperimentResults, analyse, load_human_codes
from .design import Item, Trial, load_evidence, load_items, trials
from .run import ExperimentTaskSettings, run_trials, scorer_request, segment_resolver, subject_request
from .lexical import lexical_motive, load_motive_lexicon
from .score import parse_scorer, parse_subject

__all__ = [
    "AnalysisParams", "ExperimentResults", "ExperimentTaskSettings", "Item", "Trial", "analyse",
    "lexical_motive", "load_evidence", "load_human_codes", "load_items", "load_motive_lexicon",
    "parse_scorer", "parse_subject", "run_trials", "scorer_request", "segment_resolver", "subject_request",
    "trials",
]
