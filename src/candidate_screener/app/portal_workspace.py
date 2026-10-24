"""Join CSV and operational jobs without changing either source's rankings."""
from __future__ import annotations

from .portal_jobs import JobService
from .portal_ranking import RankingService


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
    return controller
