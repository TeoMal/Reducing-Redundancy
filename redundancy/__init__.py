"""Reducing Redundancy in Neural Network Training.

A small, dataset-agnostic framework for experimenting with redundancy-reduction
techniques during neural-network training, as studied in the BSc thesis
*Reducing Redundancy in Neural Network Training* (T. Mallios, NKUA, 2025).

The package is organised around three pluggable pieces:

* ``redundancy.datasets`` -- one subpackage per dataset, each bundling its data
  loaders, normalisation statistics and the architectures appropriate for it.
* ``redundancy.methods``  -- the redundancy-reduction techniques themselves
  (loss-based example selection and low-rank gradient approximation), written
  to be independent of any particular dataset or model.
* ``redundancy.engine``   -- a minimal training/evaluation loop that wires a
  dataset, a model and a method together.
"""

__version__ = "0.1.0"
