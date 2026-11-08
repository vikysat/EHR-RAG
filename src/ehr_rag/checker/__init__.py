"""Deterministic citation checker and enforcement (rewrite / filter of failing sentences).

Kept outside ``ehr_rag.eval`` on purpose: evaluation never imports this package, so the
checker can never grade the output it enforced (tested).
"""
