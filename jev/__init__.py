"""Meno-JEV: loglikelihood (JEV-style) answer scoring over an LLM.

The framework turns any causal language model into a *scorer*: instead of
generating text, it computes the probability of each candidate answer given the
context (state), the question (instruction) and the answer space (criteria),
then returns a distribution over candidates.

Layout
------
- ``jev.data``   : benchmark parsing (state / instruction / criteria / gold).
- ``jev.models`` : model adapters. One file per model family, self-registering.
- ``jev.scoring``: loglikelihood maths (continuation / next-token, softmax).
- ``jev.tasks``  : per-source "recipes" that turn an item into a scorable query.
- ``jev.eval``   : the runner, metrics and reporting.
- ``jev.cli``    : command line entry point.
"""

__version__ = "0.1.0"
