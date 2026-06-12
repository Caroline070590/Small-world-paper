"""smallworld_qtn: small-world topology from time-series representations.

Core API for the paper *Small-World Organization as a Representation-Dependent
Signature of Nonlinear Dynamics in Biological Time Series*.

Subpackages / modules
---------------------
* :mod:`smallworld_qtn.representations`  -- QG/QTN, GAF, MTF transforms.
* :mod:`smallworld_qtn.network_metrics`  -- thresholding + small-world metrics.
* :mod:`smallworld_qtn.pipeline`         -- cleaned signal -> representation -> metrics.
* :mod:`smallworld_qtn.preprocessing`    -- per-dataset signal cleaning.
"""

from . import representations, network_metrics, pipeline

__all__ = ["representations", "network_metrics", "pipeline"]
__version__ = "0.1.0"
