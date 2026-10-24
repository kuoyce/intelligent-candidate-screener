import io
import json
import zipfile
import pytest
from candidate_screener.app.portal_reports import build_report, build_package


def snapshot(count=10):
    return {"snapshot_id":"synthetic", "candidates":[
        {"rank":i+1,"candidate_id":f"../../candidate-{i}","score":.9-i*.01,
         "cv_text":"Skills: Python. Contact private@example.com", "profile":{"TITLE":"Developer"}}
        for i in range(count)]}


def test_report_copies_ranks_and_deterministic_safe_package():
    source=snapshot()
    report=build_report(source)
    assert report["complete"] and len(report["comparison"])==10
    package=build_package(source,report)
    assert package==build_package(source,report)
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        names=archive.namelist()
        assert len([name for name in names if name.startswith("cvs/")])==10
        assert not any(".." in name for name in names)
        assert all(b"private@example.com" not in archive.read(name) for name in names)
        manifest=json.loads(archive.read("manifest.json"))
        assert len(manifest["attachments"])==10 and "PDF" in manifest["format"]


@pytest.mark.parametrize("count",[7,8,9])
def test_partial_pool_report_exists_but_package_blocked(count):
    source=snapshot(count)
    report=build_report(source)
    assert not report["complete"]
    with pytest.raises(ValueError,match="ten ranked"):
        build_package(source,report)


def test_missing_source_and_stale_snapshot_block_package():
    source=snapshot()
    report=build_report(source)
    source["candidates"][0]["score"]=.1
    with pytest.raises(ValueError,match="different ranking snapshot"):
        build_package(source,report)
    source["candidates"][0]["cv_text"]=None
    with pytest.raises(ValueError,match="source CV unavailable"):
        build_package(source,build_report(source))


def test_html_injection_and_modified_report_rejected():
    source=snapshot()
    source["candidates"][0]["profile"]["TITLE"]="<script>alert(1)</script>"
    report=build_report(source)
    assert "<script>" not in report["html"]
    report["html"]="private@example.com"
    with pytest.raises(ValueError,match="deterministic redacted"):
        build_package(source,report)


def test_report_does_not_sort_or_backfill_invalid_rank_order():
    source=snapshot()
    source["candidates"][1]["rank"]=3
    with pytest.raises(ValueError,match="unchanged contiguous"):
        build_report(source)


def test_real_portal_dataset_snapshot_public_views_reach_report_and_package():
    from candidate_screener.app.portal_data import PortalDataset
    source=snapshot()
    payload={"candidates":source["candidates"],"jd_entities":{"KNOWLEDGE":["Python"]},
             "jd_text":"Python developer role", "source_flags":{}}
    stored={"snapshot_id":"synthetic","run_id":"run","jd_id":"JD1","jd_title":"Developer",
            "digest":"digest","_payload_json":json.dumps(payload)}
    dataset=PortalDataset("run","digest",json.dumps([{"jd_id":"JD1"}]),json.dumps({"JD1":stored}))
    real=dataset.snapshot("JD1")
    report=build_report(real)
    assert report["complete"] and len(report["comparison"])==10
    assert "JD model mentions" in report["text"]
    assert len(report["top8_ids"])==8
    package=build_package(real,report)
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        assert len([name for name in archive.namelist() if name.startswith("cvs/")])==10
        assert b"_payload_json" not in archive.read("comparison-report.txt")


def test_backend_decimal_score_survives_redacted_report_and_package_exactly():
    source=snapshot()
    source["candidates"][0]["score"]=0.359416827319284
    source["candidates"][0]["profile"]["note"]="Contact private@example.com"
    source["candidates"][0]["table_aggregates"]={"role_prob":0.9123456789123}
    report=build_report(source)
    assert str(source["candidates"][0]["score"]) in report["text"]
    assert "0.9123456789123" in report["text"]
    assert "private@example.com" not in report["text"]
    with zipfile.ZipFile(io.BytesIO(build_package(source,report))) as archive:
        text=archive.read("comparison-report.txt").decode()
        assert text==report["text"]
        assert "0.359416827319284" in text


def test_generated_identity_numeric_runs_survive_report_zip_and_manifest():
    source=snapshot()
    source["snapshot_id"]="a"+"1234567890"*6+"abc"
    for index,row in enumerate(source["candidates"]):
        row["candidate_id"]=f"cv-{'1234567890'*6}{index:04d}"
        row["source_digest"]="b"+"1234567890"*6+"abc"
    report=build_report(source)
    assert source["snapshot_id"] in report["text"]
    assert report["comparison"][0]["candidate_id"]==source["candidates"][0]["candidate_id"]
    with zipfile.ZipFile(io.BytesIO(build_package(source,report))) as archive:
        manifest=json.loads(archive.read("manifest.json"))
        assert manifest["snapshot_id"]==source["snapshot_id"]
        assert [item["candidate_id"] for item in manifest["attachments"]]==report["candidate_ids"]
        assert manifest["attachments"][0]["source_digest"]==source["candidates"][0]["source_digest"]
        assert source["snapshot_id"] in archive.read("comparison-report.txt").decode()


def test_identity_redaction_bypass_does_not_apply_to_source_prose():
    source=snapshot()
    source["candidates"][0]["profile"]={"note":"Contact private@example.com"}
    assert "private@example.com" not in build_report(source)["text"]


def test_v2_contains_tenth_cv_profile_and_jd_context_while_v1_stays_eight():
    source=snapshot()
    source.update(jd_text="Python role. Contact private@example.com",jd_title="Developer")
    for version, count in ((1,8),(2,10)):
        report=build_report(source,schema_version=version)
        with zipfile.ZipFile(io.BytesIO(build_package(source,report))) as archive:
            manifest=json.loads(archive.read("manifest.json"))
            assert manifest["schema_version"]==version
            assert len(manifest["attachments"])==count
            assert len([n for n in archive.namelist() if n.startswith("profiles/")])==count
            if version==2:
                assert b"Python role" in archive.read("jd-context.json")
                assert b"private@example.com" not in archive.read("jd-context.json")
                assert manifest["attachments"][-1]["rank"]==10


def test_v2_requires_source_for_ninth_and_tenth_candidates():
    source=snapshot()
    source["candidates"][9]["cv_text"]=None
    with pytest.raises(ValueError,match="rank 10"):
        build_package(source,build_report(source))


def test_legacy_printable_report_retains_eight_shortlist_and_new_report_has_ten():
    source=snapshot()
    report=build_report(source,schema_version=1)
    section=next(line for line in report["text"].splitlines() if line.startswith("Proposed shortlist"))
    assert all(identifier in section for identifier in report["top8_ids"])
    assert source["candidates"][8]["candidate_id"] not in section
    assert source["candidates"][9]["candidate_id"] not in section
    assert "Proposed shortlist ranks 1–8:" in report["html"]
    partial=build_report(snapshot(7),schema_version=1)
    assert "incomplete: 7 of 8" in partial["text"]
    assert "handoff remains blocked" in partial["text"]
    current=build_report(source)
    assert len(current['top8_ids'])==8 and len(current['shortlist_ids'])==10
    assert 'Proposed shortlist ranks 1–10:' in current['text']


def test_unversioned_historical_report_can_still_build_eight_cv_package():
    source=snapshot();report=build_report(source,schema_version=1)
    report.pop('schema_version');report.pop('shortlist_ids')
    with zipfile.ZipFile(io.BytesIO(build_package(source,report))) as archive:
        assert json.loads(archive.read('manifest.json'))['schema_version']==1
        assert len([n for n in archive.namelist() if n.startswith('cvs/')])==8


@pytest.mark.parametrize('version',[0,3,True,'2'])
def test_report_rejects_unknown_or_untyped_schema_version(version):
    with pytest.raises(ValueError,match='schema version'):
        build_report(snapshot(),schema_version=version)


def test_v2_profiles_retain_backend_values_and_redact_entity_prose():
    source=snapshot()
    source['candidates'][-1]['entities']={'SKILL':['Python','Contact private@example.com']}
    source['candidates'][-1]['score']=0.359416827319284
    report=build_report(source)
    with zipfile.ZipFile(io.BytesIO(build_package(source,report))) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        profile=json.loads(archive.read(manifest['attachments'][-1]['profile_path']))
        assert profile['rank']==10 and profile['score']==source['candidates'][-1]['score']
        assert profile['entities']['SKILL']==['Python','Contact [EMAIL]']


@pytest.mark.parametrize('semantics',[
    'Cosine distance; smaller values are closer',
    'Calibrated probability of Fit from the backend calibration set',
])
def test_v2_live_score_meaning_survives_report_comparison_and_profiles(semantics):
    source=snapshot()
    source['source_flags']={'source_type':'operational_live','score_semantics':semantics}
    for row in source['candidates']:row['score_semantics']=semantics
    report=build_report(source)
    assert all(row['score_semantics']==semantics for row in report['comparison'])
    assert semantics in report['text'] and semantics in report['html']
    assert 'score meaning is supplied by the backend' in report['text']
    assert 'not Fit probabilities' not in report['text']
    assert [row['score'] for row in report['comparison']]==[row['score'] for row in source['candidates']]
    with zipfile.ZipFile(io.BytesIO(build_package(source,report))) as archive:
        comparison=json.loads(archive.read('comparison.json'))
        assert all(row['score_semantics']==semantics for row in comparison)
        manifest=json.loads(archive.read('manifest.json'))
        for entry in manifest['attachments']:
            assert json.loads(archive.read(entry['profile_path']))['score_semantics']==semantics


def test_v1_report_text_keeps_historical_format_when_backend_semantics_are_present():
    source=snapshot();source['source_flags']={'source_type':'operational_live'}
    for row in source['candidates']:row['score_semantics']='Calibrated probability of Fit'
    report=build_report(source,schema_version=1)
    expected=[
        'Candidate review report','Snapshot: synthetic','JD: Unavailable','Backend: Unspecified',
        'Comparison: complete top 10',
        'Ranks and scores are copied from the backend; scores are not Fit probabilities.',
        'Role family is not a verified job title.',
        'Model table fields without source offsets are unverified mentions, not source citations.',
        'Proposed shortlist ranks 1–8: '+', '.join(row['candidate_id'] for row in source['candidates'][:8]),
    ]
    expected += [f"Rank {row['rank']} | {row['candidate_id']} | score {row['score']} | {{\"TITLE\": \"Developer\"}} | table aggregates {{}}"
                 for row in source['candidates']]
    assert report['text']=='\n'.join(expected)


def test_v2_csv_keeps_existing_score_disclaimer_and_redacts_semantics():
    source=snapshot()
    for row in source['candidates']:row['score_semantics']='X2 ordering score; not a Fit probability. Contact private@example.com'
    report=build_report(source)
    assert 'scores are not Fit probabilities.' in report['text']
    assert report['comparison'][0]['score_semantics'].endswith('Contact [EMAIL]')
    assert 'private@example.com' not in report['text']
