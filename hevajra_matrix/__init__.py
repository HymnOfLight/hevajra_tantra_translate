"""hevajra_matrix: validated, evidence-graded collation of the Hevajratantra witnesses.

Rows are reference units, columns are witnesses, and each cell records how a witness
renders a unit, graded by the evidence behind it. Claude Opus 5.5 proposes, code
verifies, humans decide, and no rate is reported before the instrument is validated.
See docs/architecture.md for the module map.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hevajra-matrix")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+unknown"
