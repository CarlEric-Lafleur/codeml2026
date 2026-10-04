"""The production model of baseline_model.ipynb, replicated through our pipeline.

It sits in the ledger as the model we are compared against. It is never the retained spec
and never sent to the scorer.
"""
from equialgo.pipeline import BASELINE_SPEC

SPEC = dict(BASELINE_SPEC)
