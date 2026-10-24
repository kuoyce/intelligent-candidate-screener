import pytest
from candidate_screener.entity_annotation.text import prepare_text, map_selection, python_to_utf16, utf16_to_python


def test_mask_length_changes_and_emoji_map_exact_source():
    original = "😀 mail very.long@example.com then Python"
    prepared = prepare_text(original)
    display = prepared["display_text"]
    start = display.index("Python")
    a, b = map_selection(prepared, python_to_utf16(display, start), python_to_utf16(display, start + 6))
    assert original[a:b] == "Python"
    assert a == original.index("Python")
    assert "very.long" not in display
    assert len(prepared["boundary_map"]) == len(display) + 1


def test_mask_intersection_rejected_even_whole_display():
    prepared = prepare_text("Python person@example.com engineer")
    with pytest.raises(ValueError, match="masked"):
        map_selection(prepared, 0, python_to_utf16(prepared["display_text"], len(prepared["display_text"])))


@pytest.mark.parametrize("offset", [1, -1, True, 99])
def test_bad_utf16_boundaries_rejected(offset):
    with pytest.raises(ValueError):
        utf16_to_python("😀a", offset)


def test_multiple_sequential_redactions_keep_trailing_offsets():
    original = "https://site.example user@example.com +380 123 456 789 123456789 SQL"
    prepared = prepare_text(original)
    display = prepared["display_text"]
    a = display.index("SQL")
    start, end = map_selection(prepared, a, a + 3)
    assert original[start:end] == "SQL"
    assert len(prepared["boundary_map"]) == len(display) + 1
