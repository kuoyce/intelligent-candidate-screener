# Entity boundary guide (entity-guide-v1)

Campaign category selection remains pending. The following definitions apply only
to categories explicitly selected in the frozen campaign; do not annotate other
categories or interpret their absence as negative evidence.

| Category | Include | Exclude / boundary |
|---|---|---|
| KNOWLEDGE | Named technologies, languages, software and platforms: Python, C++, PostgreSQL | Generic abilities; preserve punctuation and informative product/version qualifiers |
| SKILL | Explicit professional competencies: project management, stakeholder communication | A technology alone belongs to KNOWLEDGE; do not infer abilities from a job title |
| TITLE | Explicit occupations or job titles: Senior Software Engineer | Study disciplines, employer names and department headings |
| DEGREE | Explicit academic qualification: Bachelor of Science in Computer Science | Bare study discipline, school name, attendance dates or course duration |
| EXPERIENCE_YEARS (only if selected) | Explicit duration tied to professional work, such as "3 years" in "3 years of accounting experience" | Education/course duration, age, calendar dates, inferred years or a number without professional-experience context |

Select the maximal informative mention with no surrounding punctuation or whitespace.
Use one category for each span; no nested, overlapping or duplicate spans. A
technology mention inside an inseparable competency should use the most specific
explicit category; if the rule does not resolve it, mark uncertain for human
adjudication rather than inventing a second overlapping annotation. Annotate
coordinated independent entities separately ("Python and SQL" gives two spans).
Annotate every repeated mention. Negated or desired qualifications remain explicit
mentions; these labels do not assert possession or proficiency. Dates and educational
durations are contextual, never automatic job-title evidence.

Read the whole document, add all missing mentions, correct boundaries/categories
and delete false spans before selecting reviewed. Empty reviewed documents are
valid only after a complete read. In-progress, skipped and uncertain documents
are not gold negatives. Masked identifiers cannot be annotated. Never reconstruct
hidden PII. Browser offsets are mapped to original text by the service; annotators
must not edit text to force alignment.

Dev/test starts blank and remains blind to model output, scores and pair Fit labels.
Train suggestions may appear only after an independent first pass and retain
separate provenance. A suggestion is not accepted human gold until reviewed.
Guideline changes require a version increment and reannotation of affected pilot
examples; calibration examples remain train only. Keep each person's revisions
separate, and resolve disagreements through a separately recorded human decision.
