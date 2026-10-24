from candidate_screener.llm.qa_list import list_windows, is_list_question


def test_separate_qualification_sentences_get_independent_exact_contexts():
    quote = "Holds an AWS certification. Holds an Azure certification."
    windows = list_windows("What certifications does the applicant hold?", {"quote": quote})
    assert len(windows) == 2
    assert any("AWS" in text and "Azure" not in text for text, _ in windows)
    assert any("Azure" in text and "AWS" not in text for text, _ in windows)
    assert all(quote[offset:offset + len(text)] == text for text, offset in windows)
    assert not windows.incomplete and not windows.exhaustive


def test_heading_bullet_section_kept_until_next_heading_without_invented_names():
    quote = "CERTIFICATIONS:\n- Qualification One\n- Qualification Two\nEXPERIENCE:\nBuilt software."
    windows = list_windows("List all certificates", {"quote": quote})
    assert any(text.startswith("CERTIFICATIONS:") and "Qualification Two" in text and "EXPERIENCE" not in text for text, _ in windows)
    assert any("Qualification Two" in text and "Qualification One" not in text for text, _ in windows)
    assert all(quote[offset:offset + len(text)] == text for text, offset in windows)


def test_expired_negated_and_ongoing_context_never_removed_or_relabelled():
    quote = "Held a qualification. It expired in2022.\nNo current certification.\nPursuing another degree."
    windows = list_windows("What qualifications does the applicant hold?", {"quote": quote})
    assert any("Held a qualification." in text and "It expired" in text for text, _ in windows)
    assert any("No current certification" in text for text, _ in windows)
    assert any("Pursuing another degree" in text for text, _ in windows)


def test_unicode_offsets_are_quote_local_python_not_utf16():
    quote = "😀 Intro.\nHolds a qualification.\nHolds a degree."
    windows = list_windows("List degrees", {"quote": quote, "passage_id": "synthetic"})
    text, offset = next((text, offset) for text, offset in windows if text.startswith("Holds a degree"))
    assert offset == quote.index("Holds a degree")
    assert quote[offset:offset + len(text)] == text


def test_window_cap_explicitly_discloses_incomplete_coverage():
    quote = "\n".join(f"Qualification entry {i}." for i in range(40))
    windows = list_windows("List qualifications", {"quote": quote})
    assert len(windows) == 32 and windows.incomplete and windows.total_windows == 40
    assert not windows.exhaustive


def test_nonlist_and_unknown_text_do_not_create_qualifications():
    quote = "No information is supplied."
    assert not is_list_question("When did the applicant move?")
    assert list_windows("When did the applicant move?", {"quote": quote}) == [(quote, 0)]
    windows = list_windows("What certifications are recorded?", {"quote": quote})
    assert windows == [(quote, 0)]
    assert not windows.exhaustive
