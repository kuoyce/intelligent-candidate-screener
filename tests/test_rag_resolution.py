"""Candidate resolution invariants, using synthetic stable document IDs."""
from types import SimpleNamespace

from candidate_screener.app.rag_service import _references, RagService


def rows():
    return [SimpleNamespace(cv_id=f"Applicant-{letter}", rank=i) for i, letter in enumerate("ABCD", 1)]


def test_full_stable_ids_are_not_reparsed_as_alias_suffixes():
    assert _references("What certifications does Applicant-D hold?", rows()) == ("Applicant-D",)
    assert _references("Compare Applicant-A and Applicant-B", rows()) == ("Applicant-A", "Applicant-B")


def test_plural_candidate_filter_has_no_phantom_s_alias():
    assert _references("Only consider candidates with AWS certification", rows()) == ()


def test_comparison_dates_are_not_candidate_ranks():
    assert _references("Compare Applicant-A and Applicant-B on certifications obtained in 2024", rows()) == ("Applicant-A", "Applicant-B")
    assert _references("Compare 1 and 2 on certifications obtained in 2024", rows()) == ("Applicant-A", "Applicant-B")


def test_unknown_explicit_reference_still_rejected():
    assert _references("Compare Applicant-A and candidate 99", rows()) is None


def test_short_candidate_id_a_is_not_indefinite_article():
    candidates = [SimpleNamespace(cv_id=letter, rank=i) for i, letter in enumerate("AB", 1)]
    assert _references("Does B have a degree?", candidates) == ("B",)
    assert _references("Does candidate a have a degree?", candidates) == ("A",)


def test_filter_does_not_infer_aws_certification_from_unrelated_azure_qualification():
    snapshot = SimpleNamespace(snapshot_id="synthetic", jd_text="AWS role", results=tuple(rows()),
        candidate_texts={"Applicant-A": "AWS experience. Holds an Azure certification.",
                         "Applicant-B": "Holds AWS certification. No Java experience.",
                         "Applicant-C": "AWS certification expired in 2024.",
                         "Applicant-D": "No AWS certification."})
    assert RagService(snapshot).filter_preview("AWS certification") == ("Applicant-B",)
