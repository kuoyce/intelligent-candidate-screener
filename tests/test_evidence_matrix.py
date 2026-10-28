"""Matrix flags are ready before generation and retain source ownership."""
from test_conversation_v3 import Composer, service, snapshot


def test_matrix_and_broad_table_exist_before_single_composition_attempt():
    current = snapshot()
    engine = service(None)
    compare = engine.reader.base.compare
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return compare(*args, **kwargs)

    class CheckingComposer(Composer):
        def compose_before(self, question, frame, facts, deadline):
            assert len(calls) == 1
            assert len(facts["evidence_matrix"]) >= 7
            assert facts["evidence_matrix"][3]["statuses"]["CV-B"] == "Qualified"
            assert facts["evidence_matrix"][2]["statuses"]["CV-A"] == "Not Found"
            assert facts["matrix_relations"]
            return super().compose_before(question, frame, facts, deadline)

    model = CheckingComposer("Candidate 1 has completed a Bachelor degree, while Candidate 2 is pursuing a Master degree.")
    engine.composer = model
    engine.reader.base.compare = counted
    answer = engine.answer(current, "Compare candidates 1 and 2.")
    assert len(calls) == 1
    assert answer["generation_mode"] == "qwen_prose"
    assert len(model.calls) == 1


def test_matrix_only_source_quotes_stay_out_of_selected_passages():
    current = snapshot()
    current.candidates[0]["preview_text"] += " Python development skills. AWS certification expired in 2021."
    model = Composer("unsupported")
    service(model).answer(current, "Compare candidates 1 and 2 on qualifications")
    bundle = model.calls[0][2]
    assert bundle["evidence_matrix"]
    assert any(row["criterion"] == "Certification" and row["statuses"]["CV-A"] == "Qualified"
               for row in bundle["evidence_matrix"])
    assert all("expired" not in item["context"] for item in bundle["evidence"])


def test_matrix_claims_keep_qualification_and_correct_owner():
    from candidate_screener.app.conversation_context import validate_prose
    model = Composer("unsupported")
    service(model).answer(snapshot(), "Compare candidates 1 and 2 on qualifications")
    bundle = model.calls[0][2]
    assert bundle["matrix_assertions"]
    assert validate_prose("Candidate 2 has completed a Master degree.", bundle)[1] == "unsupported_paraphrase"
    assert validate_prose("Candidate 1 is pursuing a Master degree.", bundle)[1] == "unsupported_paraphrase"
    accepted = "Candidate 1 has completed a Bachelor degree, while Candidate 2 is pursuing a Master degree."
    assert validate_prose(accepted, bundle)[1] is None


def test_partial_matrix_never_asserts_positive_completion_relations():
    current = snapshot()
    current.candidates[0]["preview_text"] = "Completed Bachelor degree. " + "x" * 8000 + "."
    model = Composer("unsupported")
    service(model).answer(current, "Compare candidates 1 and 2 on qualifications")
    bundle = model.calls[0][2]
    assert all(row["statuses"]["CV-A"] == "Partial" for row in bundle["evidence_matrix"])
    assert not any("completed" in item["text"] for item in bundle["matrix_assertions"]
                   if "CV-A" in item["owners"])
    assert not any("different qualification completion" in item["text"] for item in bundle["matrix_relations"])


def test_directed_matrix_namespace_and_numeric_criterion_are_preserved():
    model = Composer("unsupported")
    service(model).answer(snapshot(), "Why is candidate 1 better than candidate 2 on qualifications?")
    bundle = model.calls[0][2]
    assert all(not identity.startswith("matrix:matrix:") for item in bundle["matrix_assertions"]
               for identity in item["cell_ids"])
    assert all(item["evidence_ids"] for item in bundle["matrix_relations"])
    service(model).answer(snapshot(), "Compare candidates 1 and 2 on 5 years of Python experience")
    assert any("5 Years" in row["criterion"] for row in model.calls[-1][2]["evidence_matrix"])


def test_matrix_cannot_borrow_another_candidates_source_reference():
    from candidate_screener.app.conversation_context import build_context
    current = snapshot()
    answer = service(None).answer(current, "Compare candidates 1 and 2 on qualifications")
    degree = answer["comparison_rows"][3]
    degree["answers"]["CV-A"].update(
        evidence_ids=degree["answers"]["CV-B"]["evidence_ids"],
        text="Master degree in progress.", context="Master degree in progress.")
    bundle = build_context(current, answer)
    assert bundle["evidence_matrix"][3]["statuses"]["CV-A"] == "Not Found"
    assert bundle["evidence_matrix"][3]["evidence_ids"]["CV-A"] == []
    assert not any("Candidate 1 is pursuing" in item["text"] for item in bundle["matrix_assertions"])
