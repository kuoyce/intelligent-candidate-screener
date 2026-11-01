# User guide: start the Hiring Desk and use it

A rough walkthrough, from the build to a first chat answer. It assumes these steps are done
(README, "Set up a fresh clone"):

1. the repository is cloned;
2. the owner has handed over the data folder to `data/datapack/`;
3. you are in the clone, `logs/` exists, and `.env` was copied from `.env.example`.

The Docker path needs only Docker and git: no Python or uv on this machine. Give Docker about
15 GB of memory and 30 GB of free disk.

## 4. Build the images

```bash
docker compose build
```

- **What to expect:** two images, `app` and `llm4b` (Qwen3.5-4B). The output ends with `app  Built`
  and `llm4b  Built`.
- **How long:** a first build downloads about 7.5 GB, including the 2.74 GB model, and takes
  several minutes. A rebuild with nothing changed takes seconds.
- **Lighter setup** (a machine that cannot give Docker 15 GB): set `HR_PORTAL_CHAT_VERSION=v1` in
  `.env` and run `docker compose build app` instead. Chat is then extractive only (v1).

## 5. Check the data folder

```bash
docker compose --profile tools run --rm tools verify
```

**Expect** the last line: `OK: 24 inputs (256.8 MB) match the manifest and what the code expects`.
Anything else means the data folder is incomplete. Ask the owner for a new handover.

## 6. Start

```bash
docker compose up -d --wait
```

```bash
docker compose ps
```

- **Expect** `app` and `llm4b` both `Up … (healthy)`, after about 20 s.
- **Lighter setup:** run `docker compose up -d app`.
- **Errors:**
  - "port is already allocated": something else uses 8503. Set `HIRING_DESK_PORT` in `.env`.
  - `stat …compose.llm4b.yaml`: delete the `COMPOSE_FILE` line from `.env`.

## 7. Open the app and sign in

Open **http://localhost:8503**.

1. **First time:** the page is headed **Create the first account**. Fill in:
   - **Login ID**, for example `reviewer`;
   - **Departments**: leave `benchmark`;
   - **Access token**, typed twice: 24-512 characters, your own;
   - the expiry: up to 40 days.

   Click **Create the first account**. *Expect:* "The first account was created".
2. **Local reviewer sign-in:** enter the login ID and token, then click **Sign in**.

   *Expect:* the left sidebar shows "Signed in: …" and the **Navigation** menu, with **Review
   jobs** open.

Later accounts are made from a terminal. The command asks for the token twice, hidden:

```bash
docker compose run --rm app python -m candidate_screener.app.portal_auth --create-local-user <login-id> --department benchmark
```

## 8. Review a job (Navigation → Review jobs)

1. **Find your next review** shows the job cards (60 benchmark JDs).
   - Use **Search job titles** to filter them, or **Previous JD** / **Next JD** to page through.
   - Click a card to select it.
2. *Expect:* **JD summary** (its skills and experience), the **Full job description**, then
   **Candidate review** with the 10 ranked applicants. **Ranking details** explains the scores.
3. **Review action** menu:

   | Choose | What happens |
   |---|---|
   | **Top10 ranked CVs** | A table of rank, applicant and score |
   | **Open CV** | Pick an applicant; the redacted CV text opens |
   | **Generate session report** | A text report you can download |
   | **Send to hiring department** | Pick a department, then **Preview email** and **Send to demo outbox**. This is a local demo; no email is sent |
   | **Review another job** | Back to the job cards |

4. **Find jobs by requirement** (above the cards): type a requirement, for example "Jira", and click
   **Search requirements**. Matching JD passages are listed; your selected review does not change.

## 9. Ask Hiring Desk (the chat, below each review)

The caption reads **Hiring Desk v3**. Type in **Ask about the selected JD, applicants or supporting
evidence** and press Enter. Try these in order:

| Ask | Expect |
|---|---|
| `What does this role require?` | The JD's requirements, quoted |
| `What degree does candidate 1 have?` | The degree with its status (completed / in progress), or "can't confirm" |
| `Compare candidates 1 and 2` | A comparison table and a short explanation |
| `What about their qualifications?` | A follow-up on the same two candidates |
| `Why is 1 ranked above 2?` | One answer line, its supporting points and any counterpoints |
| `Summarise why candidate 1 is a good fit to this JD` | Strengths with quotes, and the requirements not confirmed |

- **Timing:** each answer takes up to about a minute. The 4B runs on the CPU, with a 60 s limit per
  turn.
- **Under each answer:**
  - **Show supporting sources** opens the quoted passages.
  - **Answer details** shows who wrote it. `qwen_prose` means the 4B's text was used.
    `qwen_fallback` means the app's own checked wording was used, and `fallback_reason` says why:
    for example `deadline_exceeded` (too slow) or `unsupported_paraphrase` (the 4B's wording failed
    the checks). Fallbacks are normal, especially while a ranking runs.
- **The chat will not** change ranks or scores, or decide Fit. A missing mention is not proof that
  a candidate lacks something.

## 10. Rank a new job (Navigation → Create new JD)

1. Click **Start a new JD**. Answer the **Guided requirements** one by one: title, department,
   responsibilities, skills, experience, qualifications. Click **Save answer** for each, or **Leave
   this requirement unspecified**.
2. Check the **Full JD preview**, click **Confirm JD preview**, then **Publish JD**.
3. Click **Rank candidates**. *Expect:* "Ranking status: ready" after about 15-45 s. The JD is
   ranked against the 600 A2 CVs.
4. Optional: tick **Refine the top 50 with the reranker (D64, a few minutes)** before ranking. It
   runs in the background for about 2 minutes. Click **Refresh ranking status** to collect it.
5. Click **Review this JD**, and review it as in step 8.

## 11. Alerts and saved reviews

- **Navigation → Alerts:** JD publications, rankings ready or failed, and captured emails. Click
  **Refresh alerts** to update.
- **Navigation → Saved reviews:** reopen a review you worked on. The chat and the selected
  candidates are kept per job.
- **Log out** is at the bottom of the sidebar.

## 12. Stop and restart

```bash
docker compose stop
```

```bash
docker compose up -d --wait
```

Accounts and reviews persist in `data/datapack/processed/hr-portal/`. To remove the containers,
run `docker compose down`.

More detail: [`docs/hiring-desk.md`](hiring-desk.md) (every screen and setting) and
[`docs/docker.md`](docker.md) (requirements, backups, troubleshooting).
