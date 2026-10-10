"""Human review workflow: sampling plans, review sheets and committed verdicts.

    sampling   ReviewParams (run.yaml: review), VerificationSpec / AuditSpec
               (preregistration.yaml), draw_verification, draw_audit, queue, coverage,
               write_plan / read_plan
    sheets     export(items, task, out_dir, segments, cells, ...) -> sheet paths (and the
               verified glossary ``standing_renderings.csv`` beside them);
               import_(path, task, ...) -> verdicts without text; the exact sheet columns
               as constants; ``GlossaryCheck``, the callback that confirms a human
               standing_rendering (built by ``pipeline.review``, so review never imports
               the collator)
    gold_sheets, reveal_sheets, topic_sheets
               the gold, reveal and topics sheets behind ``sheets.export`` / ``import_``
    verdicts   the committed verdict format (COLUMNS), load / save / validate,
               ``decision`` and ``decision_convention``: which human decision and which
               convention (collate.v2: none, standing_rendering, transcribed_name; ""
               when not annotated) the matrix applies

Workflow (synthesis 7; no interactive tool): draw a plan -> export blind sheets -> import
blind verdicts -> export reveal sheets -> import final verdicts -> ``build`` applies them.
Sheets hold text and live under ``runs/`` only; committed verdicts hold ids and short
quotes (``data/annotations/verdicts/README.md``).

Submodules are imported explicitly (``from hevajra_matrix.review import sheets``) so that
``matrix.build`` can use ``review.verdicts`` without loading the sheet machinery.
"""
