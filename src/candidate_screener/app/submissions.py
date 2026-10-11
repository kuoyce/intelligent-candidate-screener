"""Session-only applicant review state; no model fitting or persistent transcript storage."""
from dataclasses import dataclass, field
import hashlib

DEMO_JD = "Python software engineer. Build SQL applications and perform software testing. At least 3 years of experience."
DEMO_CVS = ["Software engineer with 4 years of experience. Python, SQL and software testing. Bachelor of Science.",
            "Project manager with 2 years of experience in project management and stakeholder communication."]


def applicant_id(index: int) -> str:
    return f"Applicant-{index:03d}"


@dataclass
class Submission:
    applicant_id: str = "Applicant-001"
    written_text: str = ""
    english_text: str = ""
    segments: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    state: str = "uploaded"
    confirmed_digest: str | None = None

    def confirm(self, reviewed_text: str) -> None:
        if not reviewed_text.strip():
            raise ValueError("Review a nonempty English transcript before confirming.")
        self.english_text = reviewed_text.strip()
        self.confirmed_digest = hashlib.sha256(self.english_text.encode()).hexdigest()
        self.state = "ready"

    def scoring_text(self, mode: str = "Standalone interview") -> str:
        digest = hashlib.sha256(self.english_text.encode()).hexdigest()
        if self.state != "ready" or digest != self.confirmed_digest:
            raise ValueError("Confirm the current English transcript before screening.")
        if mode == "CV supplement":
            if not self.written_text.strip():
                raise ValueError("Add a written CV for supplement mode.")
            return f"Written CV:\n{self.written_text.strip()}\n\nReviewed interview:\n{self.english_text}"
        return self.english_text
