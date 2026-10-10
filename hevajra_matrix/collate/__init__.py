"""T1 collation: the measurement instrument (synthesis 3.1, 4.2, 4.3, 5.3).

Data flow of one replicate, every step a pure function except the client call:

    windows.plan            reference chunks + the full witness text + core window
    prompt.load_examples    the configured prompt version's examples (and, for
                            collate.v2, the verified standing renderings)
    collator.build_request  one ``LLMRequest`` per (window, replicate); no coordinates
    collator.collate        run the requests (thread pool), ``collator.parse`` each answer
    merge.verify_all        verify.verify per window (V1-V8, V11, V12), merge_chapter per
                            chapter (V9, V10), merge_text over the whole text
    consensus.consensus     k replicates -> one alignment, grades and UNALIGNED reasons
                            (conventions voted, never lowering a grade)
    consensus.agreement     replicate Fleiss kappa and link Jaccard (gate G2)
    perturb.*               placebo perturbations of a window and their scoring (G2)

``Collation`` (verify) is the verified output of one replicate: the ``Alignment`` of the
links that passed, the UNALIGNED reason of every other unit, diagnostics and
substituted-model hints. The UNALIGNED reasons cannot live in an ``Alignment`` (a
``Link`` has no status), which is why the checks return a ``Collation``.
"""

from __future__ import annotations

from .collator import (
    SCHEMA,
    RawCollation,
    TaskSettings,
    Unresolved,
    build_request,
    collate,
    parse,
    replicate_tags,
    source_name,
)
from .consensus import Agreement, Consensus, agreement, consensus
from .merge import OverlapReport, compare_overlaps, merge_chapter, merge_text, verify_all
from .prompt import (
    PROMPT_V1,
    PROMPT_V2,
    SCHEMA_V1,
    SCHEMA_V2,
    VERSIONS,
    Examples,
    load_examples,
    load_glossary,
    load_template,
    render_glossary,
    schema_for,
)
from .verify import CheckLexicon, Collation, verify
from .windows import Window, WindowParams, plan

__all__ = [
    "PROMPT_V1", "PROMPT_V2", "SCHEMA", "SCHEMA_V1", "SCHEMA_V2", "VERSIONS", "Agreement", "CheckLexicon",
    "Collation", "Consensus", "Examples", "OverlapReport", "RawCollation", "TaskSettings", "Unresolved", "Window",
    "WindowParams", "agreement", "build_request", "collate", "compare_overlaps", "consensus", "load_examples",
    "load_glossary", "load_template", "merge_chapter", "merge_text", "parse", "plan", "render_glossary",
    "replicate_tags", "schema_for", "source_name", "verify", "verify_all",
]
