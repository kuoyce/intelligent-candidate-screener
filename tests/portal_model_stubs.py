"""Explicit UI-test model substitutes: tests never depend on local weights."""
def stub_portal_models(monkeypatch):
    from candidate_screener.app import portal_entities, portal_extractive
    class EntitiesUnavailable:
        def __init__(self, *args, **kwargs):
            pass
        def prepare(self, snapshot, candidate_ids):
            return {"status": "unavailable", "model_id": "gliner-medium-fixture",
                    "documents": {owner: {"status": "unavailable", "entities": [], "coverage": {}}
                                  for owner in [snapshot.jd_id, *candidate_ids]}}
        bundle = prepare
    class QaUnavailable:
        def answer(self, question, passages):
            return {"status": "unavailable", "spans": [], "exhaustive": False}
    monkeypatch.setattr(portal_entities, "PortalEntityService", EntitiesUnavailable)
    monkeypatch.setattr(portal_extractive, "shared_qa_client", lambda: QaUnavailable())
