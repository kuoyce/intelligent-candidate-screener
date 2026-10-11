"""Explicit draft/experimental ranker selection; never silently switch models."""
from .demo import IllustrationTfidf

ILLUSTRATION = "Illustration TF-IDF (submission pool)"
X2 = "X2: extracted role + experience + TF-IDF (experimental)"
ENTITY_X2 = "Entity-weighted X2 (experimental)"
CHOICES = (ILLUSTRATION, X2, ENTITY_X2)


class RankerUnavailable(ValueError):
    pass


def make_ranker(choice, jd_text, cv_texts, *, frequencies=None, alpha=.5):
    if choice == ILLUSTRATION:
        ranker = IllustrationTfidf()
        ranker.fit([jd_text, *cv_texts])
        return ranker
    if choice not in CHOICES:
        raise ValueError("Unknown ranker selection")
    if choice == ENTITY_X2 and (frequencies is None or not frequencies.fitted):
        raise RankerUnavailable("Entity-weighted X2 requires training-only concept frequencies. "
                                "The registered development experiment is not complete locally.")
    from candidate_screener.extraction.role_family import load_role_model
    from .extracted import ExtractedRule, ExtractedPlus
    from .lexical import Tfidf
    try:
        role = load_role_model()
    except (FileNotFoundError, ImportError):
        raise RankerUnavailable("X2 requires the inherited calibrated role-model weights. "
                                "Select the illustration or restore the selected model first.") from None
    if choice == X2:
        ranker = ExtractedPlus(Tfidf(text_variant="v2"), "X2", ExtractedRule(role))
    else:
        from .entity_weighted import EntityWeightedX2
        if alpha not in (0, .25, .5, .75, 1):
            raise ValueError("Use a registered experimental alpha")
        ranker = EntityWeightedX2(alpha=alpha, extractor=frequencies.extractor,
                                  frequencies=frequencies, rule=ExtractedRule(role))
    try:
        ranker.fit(None)
    except (FileNotFoundError, ImportError):
        raise RankerUnavailable("Selected experimental ranker prerequisites are unavailable; "
                                "restore its recorded training text/cache before using it.") from None
    return ranker
