"""A Qwen-composed v2/v3 answer can be saved (plan 2026-10-27-docker, S9).

S9 found that the review store refused every answer Qwen composed: both composers wrote the
provenance key `output_token_budget`, which the store's credential rule reads as a token. These
tests run the real composers on a fake provider transport (no model) with the real provenance
format, and save the turns through PortalController.chat. Invented documents only.
"""
from __future__ import annotations

from copy import deepcopy
import json
import time

import pandas as pd
import pytest

from candidate_screener.app.conversation_service import ConversationService
from candidate_screener.app.conversation_v3 import ConversationV3Service
from candidate_screener.app.portal_controller import PortalController
from candidate_screener.app.portal_data import CV_COLUMNS, JD_COLUMNS, PortalDataset
from candidate_screener.app.portal_extractive import PortalExtractiveService
from candidate_screener.app.portal_store import PortalStore, Principal, _payload
from candidate_screener.llm.conversation_prose import ProseComposer
from candidate_screener.llm.conversation_qwen import ConversationComposer
from candidate_screener.llm.qwen import QwenConfig

JD_ID = "synthetic-jd-77"
SOURCES = [("synthetic-cv-77-1", "Completed Bachelor degree. Three years of Python experience."),
           ("synthetic-cv-77-2", "Master degree in progress. Five years of Python experience."),
           *((f"synthetic-cv-77-{rank}", f"Invented applicant {rank}. Two years of SQL experience.")
             for rank in range(3, 11))]   # a review holds exactly ten candidates
CONFIG = QwenConfig(base_url="http://127.0.0.1:8000/v1", model="Qwen/Qwen2.5-0.5B-Instruct",
                    revision="7ae557604adf67be50417f59c2c2f167def9a775", tokenizer_path="service",
                    runtime="local Transformers service", quantization="none", max_context_tokens=4096,
                    max_output_tokens=256, timeout_seconds=30., temperature=0., json_mode=False)


class Transport:
    """The provider interface the composers call, answering like Qwen without a model: v2 selects
    every offered relation; v3 returns fixed prose."""

    max_context_tokens = 4096
    max_output_tokens = 256
    config = CONFIG

    def __init__(self, prose=None):
        self.prose, self.calls = prose, 0

    def count_tokens(self, messages):
        return 300

    def complete(self, messages):
        self.calls += 1
        if self.prose is not None:
            return self.prose
        payload = json.loads(messages[-1]["content"])
        return json.dumps({"relation_ids": [relation["id"] for relation in payload["relations"]]})


class NoQa:
    def answer(self, question, passages):
        return {"status": "not_found", "spans": []}


V3_PROSE = "Candidate 1 has completed a Bachelor degree, while Candidate 2 is pursuing a Master degree."


def dataset():
    job = dict.fromkeys(JD_COLUMNS)
    job.update(jd_id=JD_ID, stratum="synthetic", jd_title="Invented data engineer", role="synthetic family",
               skills="Python", min_years=2)
    cvs, ranks = [], []
    for rank, (cv_id, _) in enumerate(SOURCES, 1):
        cv = dict.fromkeys(CV_COLUMNS)
        cv.update(cv_id=cv_id, role="synthetic family", skills="Python", years=3)
        cvs.append(cv)
        ranks.append(dict(pair_id=f"synthetic-pair-77-{rank}", stratum="synthetic", jd_id=JD_ID, cv_id=cv_id,
                          x2_score=float(10 - rank), x2_rank=rank))
    sources = {"jd": {JD_ID: "Bachelor degree required. Python experience required."}, "cv": dict(SOURCES)}
    return PortalDataset.from_folder(source_documents=sources, _is_demo=True,
                                     _frames=(pd.DataFrame(cvs), pd.DataFrame([job]), pd.DataFrame(ranks)))


def controller(tmp_path, service):
    principal = Principal("reviewer", ("demo",), time.time() + 3600, "demo")
    portal = PortalController(dataset(), PortalStore(tmp_path / "store.sqlite"), principal)
    portal.qa_service = service
    return portal


def v2_service(transport):
    return ConversationService(PortalExtractiveService(qa_client=NoQa()), composer=ConversationComposer(transport))


def v3_service(transport):
    return ConversationV3Service(PortalExtractiveService(qa_client=NoQa()), ProseComposer(transport))


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_a_qwen_composed_turn_is_saved_through_the_portal(tmp_path, version):
    transport = Transport(prose=V3_PROSE if version == "v3" else None)
    portal = controller(tmp_path, v2_service(transport) if version == "v2" else v3_service(transport))
    session = portal.open_job(JD_ID)["session_id"]
    answer = portal.chat(session, "Compare candidates 1 and 2 on qualifications", "turn-one")
    assert transport.calls == 1
    assert answer["generation_mode"] == ("qwen_answer_plan" if version == "v2" else "qwen_prose")
    assert answer["model"] == CONFIG.model
    assert answer["generation_provenance"]["revision"] == CONFIG.revision
    saved = portal.load_session(session)["state"]["qa_turns"]
    assert [turn["answer"]["generation_mode"] for turn in saved] == [answer["generation_mode"]]


def test_every_key_either_composer_writes_into_the_provenance_is_storable():
    frame = {"snapshot_id": "snapshot", "jd_id": "job", "selected_candidates": ["a", "b"],
             "intent": "comparison", "criteria": ["Python"]}
    facts = {"cells": [
        {"id": "a-python", "owner": "a", "context": "Python deployments.", "support_status": "explicit_positive"},
        {"id": "b-python", "owner": "b", "context": "No commercial Python experience.",
         "support_status": "explicit_negative"}],
        "relations": [{"id": "contrast", "kind": "explicit_contrast", "participants": ["a", "b"],
                       "cell_ids": ["a-python", "b-python"], "text": "A deploys Python; B denies it."}],
        "labels": {"a": "Candidate 1", "b": "Candidate 2"}, "history": []}
    plans = ConversationComposer(Transport()).compose("Compare", frame, deepcopy(facts))
    prose = ProseComposer(Transport(prose="Candidate 1 lists Python deployments.")).compose(
        "Compare", frame, deepcopy(facts))
    for provenance in (plans["provenance"], prose["provenance"]):
        assert {"prompt_version", "input_tokens", "elapsed_seconds", "revision"} <= set(provenance)
        for key, value in provenance.items():
            _payload({"generation_provenance": {key: value}})   # raises on a refused key
