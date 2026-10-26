"""Out-of-scope replies are exact and bypass factual extraction/generation."""
import pytest
from test_conversation_v3 import Composer, service, snapshot


@pytest.mark.parametrize("question", ["Give me a recipe of apple pie.",
    "What is the nationality of all the top 10 candidates for this JD?"])
def test_negative_cases_return_only_requested_scope_message(question):
    model = Composer("This must not be called")
    answer = service(model).answer(snapshot(), question)
    assert answer["text"] == "I am not able to answer this question. It is out of my scope."
    assert answer["status"] == "out_of_scope"
    assert answer["citations"] == []
    assert not model.calls
