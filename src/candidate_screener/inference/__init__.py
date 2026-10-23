"""The reported system as plain functions an app can call (plan 2026-10-23-inference-api).

    from candidate_screener import inference
    s = inference.Screener.load()
    res = s.rank(jd_text, cvs, jd_title=...)      # D63; rerank=True for D64; system="X2" baseline
"""
from candidate_screener.inference.model import L1Model
from candidate_screener.inference.schema import CVProfile, JDProfile, Ranked, Screening
from candidate_screener.inference.screener import Screener

__all__ = ["CVProfile", "JDProfile", "L1Model", "Ranked", "Screener", "Screening"]
