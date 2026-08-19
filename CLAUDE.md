## Project Overview

 

## Data Structure

 

## Project Tech Stack
This project will primarily use **Python** as the core programming language for data processing, analysis, and machine learning tasks. 
Use package management ‘uv’

It will support:
- Data ingestion and preprocessing of resume and job descriptions (csv, pdf, html etc..)
…

Where appropriate, commonly used Python libraries (e.g., pandas, NumPy, scikit-learn) will be leveraged to streamline development and ensure maintainability.


## Agent Workflow Guidelines

### 1. Planning Phase
Always begin by creating a detailed plan and corresponding design specifications.
- Store all plans under the `plan/` directory 
- Use the naming convention: 
    `{date}-{plan_name}`

---

 

### 2. Approval and Version Control
- Obtain confirmation/approval of the plan before proceeding 
- Once approved:
- Create a new branch 
- Commit the finalized plan to the repository 
---


### 3. Implementation Phase
- Use a sub-agent-driven development approach to implement the solution 
- Ensure implementation strictly follows the approved plan and design specifications 

---

 

### 4. Code Review and Validation
- Conduct a thorough code review after implementation 
- Verify that the implementation aligns with the approved plan and design 
- Ensure correctness, completeness, and adherence to standards 

---

### 5. Documentation Updates
- Update documentation to reflect any changes made during implementation 
This includes (but is not limited to): 
- `AGENTS.md`
- `README.md` 
- Relevant plan files 

Ensure any deviations from the original plan or design are clearly documented.

## Agent Behaviour
The agent must operate in a structured, transparent, and non-assumptive manner throughout the project lifecycle.
 

### 1. No Assumptions
- The agent must **not guess user intent, requirements, or constraints** 
- All decisions must be based on explicitly provided information 
- If any requirement, context, or ambiguity exists, the agent must:
  - Clearly highlight the uncertainty 
  - Avoid proceeding with implicit assumptions 

### 2. Clarification-First Approach
- The agent should proactively seek clarification when:
  - Requirements are incomplete or ambiguous 
  - Multiple interpretations are possible 
  - Key technical or business decisions are unspecified 
- Clarifications should be:
  - Concise and specific 
  - Organized for easy response 

### 3. Open Questions Tracking
- The agent must maintain a **persistent list of open questions** during planning and implementation 

- This list should:
  - Be explicitly documented (e.g., in plan files or as a dedicated section) 
  - Be updated as questions are resolved or new ones arise 
- No critical step should proceed without acknowledging unresolved high-impact questions 

### 4. Explicit Assumptions (When Unavoidable)
- If progress is necessary and clarification is not immediately available:
  - The agent may proceed with **clearly stated assumptions** 
  - Assumptions must be:
    - Explicitly documented 
    - Justified 
    - Highlighted for later confirmation 

### 5. Transparent Reasoning
- All key decisions, trade-offs, and design choices must be:
  - Clearly explained 
  - Traceable back to requirements or constraints 
- The agent should favour transparency over brevity when reasoning affects outcomes 

### 6. Iterative Alignment
- The agent should continuously ensure alignment with the user by:
  - Revisiting assumptions 
  - Updating open questions 
  - Flagging any deviations from the original plan 
