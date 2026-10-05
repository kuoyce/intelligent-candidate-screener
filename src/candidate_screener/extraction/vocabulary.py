"""Small offline seed lexicon; canonical IDs are explicitly not ESCO IDs."""
import hashlib
import json

VOCABULARY_VERSION = "seed-v1"
# KNOWLEDGE names technologies/tools; SKILL names competencies or activities.
VOCABULARY = {
    "KNOWLEDGE": {
        "python": ("Python",), "sql": ("SQL",), "aws": ("AWS", "Amazon Web Services"),
        "kubernetes": ("Kubernetes", "K8s"), "docker": ("Docker",),
        "javascript": ("JavaScript",), "typescript": ("TypeScript",),
        "java": ("Java",), "cpp": ("C++",), "csharp": ("C#",),
        "dotnet": (".NET",), "nodejs": ("Node.js", "NodeJS"),
        "react": ("React", "React.js"), "postgresql": ("PostgreSQL", "Postgres"),
        "git": ("Git",), "linux": ("Linux",), "go": ("Go", "Golang"), "r": ("R",),
    },
    "SKILL": {
        "machine-learning": ("machine learning",), "data-analysis": ("data analysis",),
        "project-management": ("project management",), "software-testing": ("software testing",),
        "data-engineering": ("data engineering",), "deep-learning": ("deep learning",),
    },
    "TITLE": {
        "software-engineer": ("software engineer",), "software-developer": ("software developer",),
        "data-scientist": ("data scientist",), "data-analyst": ("data analyst",),
        "data-engineer": ("data engineer",), "project-manager": ("project manager",),
        "product-manager": ("product manager",), "devops-engineer": ("DevOps engineer",),
        "frontend-developer": ("frontend developer", "front-end developer"),
        "backend-developer": ("backend developer", "back-end developer"),
    },
    "DEGREE": {
        "bachelor": ("bachelor's degree", "bachelor degree", "Bachelor of Science", "Bachelor of Arts", "BSc", "B.Sc."),
        "master": ("master's degree", "master degree", "Master of Science", "Master of Arts", "MSc", "M.Sc."),
        "phd": ("PhD", "Ph.D.", "doctorate"),
    },
}


def vocabulary_digest() -> str:
    return hashlib.sha256(json.dumps({"version": VOCABULARY_VERSION, "terms": VOCABULARY},
                                    sort_keys=True).encode()).hexdigest()
