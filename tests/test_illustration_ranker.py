import pandas as pd
import pytest

from candidate_screener.matching.demo import IllustrationTfidf


def test_illustration_uses_shared_space_and_ignores_oracle_fields():
    frame = pd.DataFrame({"pair_id": ["a", "b"], "jd_text": ["Python SQL"] * 2,
                          "cv_text": ["Python SQL", "welding plumbing"],
                          "cv_keyword": ["wrong", "Python"]})
    ranker = IllustrationTfidf()
    ranker.fit(frame)
    scores = ranker.score(frame)
    assert scores.iloc[0] == pytest.approx(1)
    assert scores.iloc[1] == pytest.approx(0)
    assert scores.equals(ranker.score(frame.assign(cv_keyword=["Python", "wrong"])))


def test_illustration_never_implicitly_loads_benchmark_data():
    with pytest.raises(ValueError, match="explicit"):
        IllustrationTfidf().fit(None)
    with pytest.raises(ValueError, match="Nonempty"):
        IllustrationTfidf().fit([""])
