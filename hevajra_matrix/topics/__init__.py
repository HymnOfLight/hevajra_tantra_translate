"""Topics: the codebook, human topic labels, their agreement, and the T3 pre-labeller.

A unit's topic group is the exposure of the sensitive-vs-neutral contrast (E4, synthesis
6.3) and defines the arms of the over-attribution experiment. Humans decide every label;
Claude (task T3, synthesis 3.3) only pre-labels, to order and pre-fill the human sheet.

    codebook   vocabulary constants, topic_group(topics), load_codebook(path)
    labels     load_labels / write_labels for data/annotations/topics/<reference>.csv,
               current_labels (fingerprint check), human_agreement (coder vs second coder
               only, critique A6) and prelabel_agreement (reported separately)
    prelabel   plan_batches, build_request, prelabel_requests, parse_prelabels (task T3)

Everything public is re-exported here, so callers write ``from hevajra_matrix.topics
import load_codebook, topic_group, ...``.

Vocabulary in code, definitions in data
    Topic names, their groups and the group precedence are pre-registered constructs, so
    they are plain constants in ``codebook`` and ``load_codebook`` refuses a codebook file
    that disagrees. Definitions, boundary notes and coding rules live only in
    ``data/codebook/topics.yaml``, which is rendered into the prompt (so editing them
    changes every cache key).

Blindness by construction
    The pre-labeller never sees the Chinese witness (synthesis 1.4): ``plan_batches``
    accepts the segments of exactly one witness written in a reference language, and a
    request holds nothing but the fixed prompt, the codebook and those segments' kinds
    and texts. Neighbouring units are shown as context that must not be labelled.

Prelabels are hints, never measurements
    A cue must occur verbatim in its unit; a non-neutral topic without a verified cue is
    dropped and flagged. Server-side fallback is on for this task (impl_decisions 4), so
    an answer served by another model is parsed as a reviewer hint but marked
    ``substituted_model``: it is never committed as ``prelabel_topics`` and therefore never
    enters an agreement statistic.
"""

from .codebook import (
    FRAME,
    GROUP_OF,
    GROUP_PRECEDENCE,
    MANTRA_CONTROL,
    NEUTRAL,
    SENSITIVE_TOPICS,
    TOPICS,
    UNLABELLED,
    CodebookError,
    TopicCodebook,
    TopicDefinition,
    TopicError,
    TopicGroup,
    load_codebook,
    topic_group,
    topic_set_error,
)
from .labels import (
    LABEL_COLUMNS,
    TOPIC_SEPARATOR,
    TopicAgreement,
    TopicKappa,
    TopicLabel,
    agreement,
    cohen_kappa,
    current_labels,
    format_topics,
    human_agreement,
    load_labels,
    parse_topics,
    prelabel_agreement,
    write_labels,
)
from .prelabel import (
    FLAG_CUE_UNVERIFIED,
    FLAG_DUPLICATE_REF,
    FLAG_NEUTRAL_NOT_ALONE,
    REFERENCE_LANGS,
    TASK,
    Prelabel,
    PrelabelBatch,
    TopicTaskSettings,
    build_request,
    parse_prelabels,
    plan_batches,
    prelabel_requests,
    prelabel_schema,
    render_body,
    render_system,
)

__all__ = [
    "FLAG_CUE_UNVERIFIED", "FLAG_DUPLICATE_REF", "FLAG_NEUTRAL_NOT_ALONE", "FRAME", "GROUP_OF",
    "GROUP_PRECEDENCE", "LABEL_COLUMNS", "MANTRA_CONTROL", "NEUTRAL", "REFERENCE_LANGS",
    "SENSITIVE_TOPICS", "TASK", "TOPICS", "TOPIC_SEPARATOR", "UNLABELLED",
    "CodebookError", "Prelabel", "PrelabelBatch", "TopicAgreement", "TopicCodebook",
    "TopicDefinition", "TopicError", "TopicGroup", "TopicKappa", "TopicLabel", "TopicTaskSettings",
    "agreement", "build_request", "cohen_kappa", "current_labels", "format_topics",
    "human_agreement", "load_codebook", "load_labels", "parse_prelabels", "parse_topics",
    "plan_batches", "prelabel_agreement", "prelabel_requests", "prelabel_schema", "render_body",
    "render_system", "topic_group", "topic_set_error", "write_labels",
]
