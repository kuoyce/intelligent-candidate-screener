"""Registry of every data source evaluated for the project.

One declaration per source drives three things: what `fetch` downloads, what
`verify` asserts is on disk, and what the dataset cards in `docs/data/cards/`
record. Figures in `expected_rows` are the **Verified** counts measured on
19 Aug 2026 and reproduced in `docs/data/data-catalog.md`; `verify` treats a
drift from them as a failure, because it means a publisher has re-uploaded.

Tiers follow the catalog: 1 = adopt, 2 = adopt with stated caveats,
3 = evaluated and rejected (kept here so the decision stays traceable).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Artifact:
    """A file group expected on disk after a successful download."""

    glob: str
    min_files: int = 1
    rows: int | None = None  # verified row count, None = record but do not assert


@dataclass(frozen=True)
class Source:
    key: str  # local id, also the directory name under data/raw/
    catalog_id: str  # A1, B2, ... as used in docs/data/data-catalog.md
    title: str
    kind: str  # hf | github | kaggle | manual
    locator: str  # HF repo id, GitHub repo, Kaggle slug, or a URL
    licence: str
    tier: int
    role: str
    pii: str
    artifacts: tuple[Artifact, ...]
    dest: str = ""  # directory under data/raw/; defaults to `key`
    adopted: bool = True  # tier 3 sources are registered but never fetched by --all
    automated: bool = True  # False = needs credentials or a manual browser step
    caveats: str = ""
    urls: tuple[str, ...] = field(default_factory=tuple)

    @property
    def directory(self) -> str:
        return self.dest or self.key


# ---------------------------------------------------------------------------
# A. Candidate–JD matching (R2 + R3)

_A1 = Source(
    key="fit",
    catalog_id="A1",
    title="cnamuangtoun/resume-job-description-fit",
    kind="hf",
    locator="cnamuangtoun/resume-job-description-fit",
    licence="None declared (accepted — decision D7)",
    tier=1,
    role="Core benchmark for Stages 1–4; substrate for the synthetic retrieval pools",
    pii="Livecareer-style resumes; names largely stripped, employers/schools remain",
    artifacts=(
        Artifact("*-train-*.parquet", rows=6_241),
        Artifact("*-test-*.parquet", rows=1_759),
    ),
    caveats="99.8% resume leakage across the shipped split; 6 conflicting-label pairs; "
            "effective size is 642+477 resumes / 280+71 JDs, not 8,000 examples",
    urls=("https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit",),
)

_A2_JD = Source(
    key="djinni-jd",
    catalog_id="A2",
    title="lang-uk/recruitment-dataset-job-descriptions-english (Djinni)",
    kind="hf",
    locator="lang-uk/recruitment-dataset-job-descriptions-english",
    licence="MIT",
    tier=1,
    role="In-domain IT job descriptions for the end-to-end evaluation set",
    pii="Pre-anonymized by the publishers",
    artifacts=(Artifact("*-train-*.parquet", rows=141_897),),
    caveats="No fit labels. Ukrainian/EE IT market — stated as a limitation (decision D11)",
    urls=("https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english",),
)

_A2_CV = Source(
    key="djinni-cv",
    catalog_id="A2",
    title="lang-uk/recruitment-dataset-candidate-profiles-english (Djinni)",
    kind="hf",
    locator="lang-uk/recruitment-dataset-candidate-profiles-english",
    licence="MIT",
    tier=1,
    role="In-domain candidate CVs; joins to the JD side on `Primary Keyword`",
    pii="Pre-anonymized by the publishers",
    artifacts=(Artifact("*-train-*.parquet", rows=210_250),),
    caveats="No fit labels; same market limitation as the JD side",
    urls=("https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english",),
)

_A4 = Source(
    key="ats",
    catalog_id="A4",
    title="0xnbk/resume-ats-score-v1-en",
    kind="hf",
    locator="0xnbk/resume-ats-score-v1-en",
    licence="Apache-2.0",
    tier=3,
    role="REJECTED — 97.2% derivative of A1; documented `[SEP]` separator absent; circular score",
    pii="Inherited from A1",
    artifacts=(Artifact("*-train-*.parquet", rows=5_099), Artifact("*-validation-*.parquet", rows=1_275)),
    adopted=False,
    caveats="Fetch only to reproduce the rejection evidence (`profile --recover-ats-boundary`)",
    urls=("https://huggingface.co/datasets/0xnbk/resume-ats-score-v1-en",),
)

# ---------------------------------------------------------------------------
# B. Information extraction / NER (R1)

_B1 = Source(
    key="dataturks",
    catalog_id="B1",
    title="DataTurks — Entity Recognition in Resumes",
    kind="github",
    locator="DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy",
    licence="None declared",
    tier=2,
    role="Resume-native NER gold seed (decision D5); de-identify before use",
    pii="DIRECT — `Name` and `Email Address` are annotated classes, plus residual emails/phones",
    artifacts=(Artifact("traindata.json", rows=200), Artifact("testdata.json", rows=20)),
    caveats="6.2% of spans carry broken offsets; only 472 Skills spans; 20 test docs is too few "
            "to carry the §7 extraction target — use cross-validation over all 220 docs",
    urls=("https://github.com/DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy",
          "https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner"),
)

_B2 = Source(
    key="skillspan",
    catalog_id="B2",
    title="jjzha/skillspan",
    kind="hf",
    locator="jjzha/skillspan",
    licence="CC-BY-4.0",
    tier=1,
    role="Primary skill/knowledge span supervision — the volume DataTurks lacks",
    pii="Pre-anonymized (entities replaced with placeholders)",
    artifacts=(
        Artifact("*-train-*.parquet", rows=4_800),
        Artifact("*-validation-*.parquet", rows=3_174),
        Artifact("*-test-*.parquet", rows=3_569),
    ),
    caveats="Annotated on job postings, not resumes — a documented domain shift",
    urls=("https://huggingface.co/datasets/jjzha/skillspan",),
)

_B3 = Source(
    key="green",
    catalog_id="B3",
    title="jjzha/green",
    kind="hf",
    locator="jjzha/green",
    licence="CC-BY-4.0",
    tier=2,
    role="Supplementary span supervision; adopt only if B2 proves insufficient",
    pii="Pre-anonymized",
    artifacts=(
        Artifact("*-train-*.parquet", rows=8_669),
        Artifact("*-validation-*.parquet", rows=964),
        Artifact("*-test-*.parquet", rows=335),
    ),
    caveats="Same annotation family as B2; green-skills focus",
    urls=("https://huggingface.co/datasets/jjzha/green",),
)

# ---------------------------------------------------------------------------
# C. Resume corpora — raw formats and clustering (R6, R7)

_C1 = Source(
    key="livecareer-pdf",
    catalog_id="C1",
    title="Kaggle snehaanbhawal/resume-dataset (PDF corpus)",
    kind="kaggle",
    locator="snehaanbhawal/resume-dataset",
    licence="Kaggle-hosted, publisher terms",
    tier=1,
    role="The only source of real PDFs — satisfies §5.2's parsing/robustness claims (decision D6)",
    pii="Livecareer-style resumes; treat as personal data even though names are largely stripped",
    dest="snehaanbhawal-resume-dataset",
    artifacts=(Artifact("Resume.csv", rows=2_484), Artifact("data/**/*.pdf", min_files=2_484)),
    automated=False,
    caveats="40 resumes (6.2%) overlap A1 and must be excluded from any NER training set or "
            "distractor pool built off this corpus",
    urls=("https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset",),
)

_C2 = Source(
    key="livecareer",
    catalog_id="C2",
    title="opensporks/resumes",
    kind="hf",
    locator="opensporks/resumes",
    licence="None declared",
    tier=1,
    role="Same corpus as C1 without the PDFs — `Resume_html` still exercises real markup parsing",
    pii="As C1",
    artifacts=(Artifact("*-train-*.parquet", rows=2_484),),
    caveats="HTML only, no PDF; no licence declared",
    urls=("https://huggingface.co/datasets/opensporks/resumes",),
)

_C3 = Source(
    key="resume-atlas",
    catalog_id="C3",
    title="ahmedheakl/resume-atlas",
    kind="hf",
    locator="ahmedheakl/resume-atlas",
    licence="MIT",
    tier=2,
    role="Clustering EDA only (R7)",
    pii="Pre-normalised text; no direct identifiers observed",
    artifacts=(Artifact("*-train-*.parquet", rows=13_389),),
    caveats="`Text` is lowercased, punctuation-stripped and stopword-removed — unusable for NER, "
            "evidence spans or anything shown in the recruiter UI",
    urls=("https://huggingface.co/datasets/ahmedheakl/resume-atlas",),
)

# ---------------------------------------------------------------------------
# D. Skill vocabulary (R5)

_D1 = Source(
    key="esco",
    catalog_id="D1",
    title="ESCO v1.2.1 classification (CSV bundle)",
    kind="manual",
    locator="https://esco.ec.europa.eu/en/use-esco/download",
    licence="European Commission ESCO terms of use",
    tier=1,
    role="Skill ontology: normalisation, alias resolution, hard vs preferred requirements (§4.5)",
    pii="None",
    dest="esco_dataset-v1.2.1-classification",
    artifacts=(
        Artifact("skills_en.csv", rows=13_960),
        Artifact("occupations_en.csv", rows=3_043),
        Artifact("occupationSkillRelations_en.csv", rows=126_051),
    ),
    automated=False,
    caveats="EU taxonomy — tech-tool coverage is thinner than a scraped tech vocabulary, hence "
            "the D2 frequency join. Redistribution of derived extracts to be confirmed (Q11)",
    urls=("https://esco.ec.europa.eu/en/use-esco/download",),
)

_D2 = Source(
    key="data-jobs",
    catalog_id="D2",
    title="lukebarousse/data_jobs",
    kind="hf",
    locator="lukebarousse/data_jobs",
    licence="Apache-2.0",
    tier=2,
    role="Tech-role skill frequencies used to weight and extend the ESCO vocabulary",
    pii="None",
    artifacts=(Artifact("*-train-*.parquet", rows=785_741),),
    caveats="Contains NO job-description text — it cannot serve as a JD corpus",
    urls=("https://huggingface.co/datasets/lukebarousse/data_jobs",),
)

SOURCES: dict[str, Source] = {
    s.key: s
    for s in (_A1, _A2_JD, _A2_CV, _A4, _B1, _B2, _B3, _C1, _C2, _C3, _D1, _D2)
}

#: Keys pulled by `fetch --all`: adopted and requiring no credentials.
AUTOMATED = [k for k, s in SOURCES.items() if s.adopted and s.automated]
#: Adopted sources that a human must place on disk (Kaggle token / registration).
MANUAL = [k for k, s in SOURCES.items() if s.adopted and not s.automated]
