"""Join CSV and operational jobs without changing either source's rankings."""
from __future__ import annotations

import os

from .portal_jobs import JobService
from .portal_ranking import RankingService

CHAT_VERSIONS = ("v1", "v2", "v3")


def chat_version():
    """The one reader of `HR_PORTAL_CHAT_VERSION`: v3 when unset (plan 2026-10-31-v3-default), v1 for an unknown value."""
    version = os.environ.get("HR_PORTAL_CHAT_VERSION", "v3").lower()
    return version if version in CHAT_VERSIONS else "v1"


class ReviewDataset:
    def __init__(self, exports, jobs, ranking):
        self.exports, self.job_service, self.ranking_service = exports, jobs, ranking
        self.is_demo = exports.is_demo

    def jobs(self):
        existing = [{**job, "source": "csv_export"} for job in self.exports.jobs()]
        operational = [{**job, "source": "live_ranking"} for job in self.job_service.list()
                       if job["status"] == "published"]
        if {job["jd_id"] for job in existing} & {job["jd_id"] for job in operational}:
            raise ValueError("Export and operational job identities overlap")
        return existing + operational

    def snapshot(self, jd_id):
        if jd_id in {job["jd_id"] for job in self.exports.jobs()}:
            return self.exports.snapshot(jd_id)
        job = self.job_service.get(jd_id)
        if job["status"] != "published":
            raise ValueError("This JD revision has not been published")
        snapshot = self.ranking_service.snapshot(jd_id)
        if snapshot is None:
            raise ValueError("This JD has no ready live ranking")
        return snapshot

    def job_documents(self):
        """JD source access never starts or requires candidate ranking."""
        documents = []
        for job in self.exports.jobs():
            snapshot = self.exports.snapshot(job["jd_id"])
            documents.append({"jd_id": snapshot.jd_id, "title": snapshot.jd_title,
                              "text": snapshot.jd_text, "source_version": snapshot.digest})
        for job in self.job_service.list():
            if job["status"] == "published":
                documents.append({"jd_id": job["jd_id"], "title": job["jd_title"],
                                  "text": job["jd_text"], "source_version": job["content_digest"]})
        return documents


def attach_services(controller):
    """Bind revocation guards once without making the guard depend on the dataset."""
    if not hasattr(controller, "job_service"):
        jobs = JobService(controller.store.path, controller.principal, guard=controller._guard)
        ranking = RankingService(controller.store.path, controller.principal, guard=controller._guard)
        controller.job_service, controller.ranking_service = jobs, ranking
        controller.dataset = ReviewDataset(controller.dataset, jobs, ranking)
    if controller.qa_service is None:
        from .portal_entities import PortalEntityService
        from .portal_extractive import PortalExtractiveService, shared_qa_client
        controller.entity_service = PortalEntityService(controller.store.path.parent / "entity-cache",
                                                        controller.principal, guard=controller._guard)
        controller.qa_service = PortalExtractiveService(controller.entity_service, shared_qa_client())
        version = chat_version()
        if version == "v3":
            from .conversation_v3 import ConversationV3Service
            from candidate_screener.llm.conversation_prose import configured_prose_composer
            composer = configured_prose_composer()
            controller.qa_service = ConversationV3Service(controller.qa_service, composer=composer,
                                                          turn_timeout=getattr(composer, "deadline_seconds", 30.))
        elif version == "v2":
            from .conversation_service import ConversationService
            from candidate_screener.llm.conversation_qwen import configured_composer
            controller.qa_service = ConversationService(controller.qa_service, composer=configured_composer())
    return controller
