"""Human review workflow: sampling plans, review sheets and committed verdicts.

    sampling   ReviewParams (run.yaml: review), VerificationSpec / AuditSpec
               (preregistration.yaml), draw_verification, draw_audit, queue, coverage,
               write_plan / read_plan
    sheets     export(items, task, out_dir, segments, cells, ...) -> sheet paths;
               import_(path, task, ...) -> verdicts without text; import_topics,
               merge_topic_labels; the exact sheet columns as constants
    verdicts   the committed verdict format (COLUMNS), load / save / validate, and
               ``decision``: which human decision the matrix applies

Workflow (synthesis 7; no interactive tool): draw a plan -> export blind sheets -> import
blind verdicts -> export reveal sheets -> import final verdicts -> ``build`` applies them.
Sheets hold text and live under ``runs/`` only; committed verdicts hold ids and short
quotes (``data/annotations/verdicts/README.md``).

Submodules are imported explicitly (``from hevajra_matrix.review import sheets``) so that
``matrix.build`` can use ``review.verdicts`` without loading the sheet machinery.
"""
