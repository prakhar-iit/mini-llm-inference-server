# mini-llm-inference-server: working rules

An LLM inference server built one optimization at a time, with a study per milestone. The roadmap is
`docs/distributed_inference_plan.md`; design decisions are in `docs/decisions/`; study notes in `docs/studies/`.

## Where things run
- **EC2 only.** The MacBook is an editor. Don't run tests, smoke runs or studies on it, and don't
  propose Mac/MPS runs.
- g6.xlarge / g6.12xlarge (L4) for development, `pytest`, smoke runs and first-pass numbers;
  p5.4xlarge (1× H100) for Phase 1 official numbers, p5.48xlarge (8× H100) for Phases 2–4, p5en for
  Kimi K2 (decision 25). One GPU per chain of studies; remeasure the baseline when the GPU changes. Label L4 numbers as first-pass and never mix them with
  official ones.
- Every paid session is scripted end to end before the instance starts (`scripts/aws/README.md`).

## Keep every doc in sync
**Any session that changes anything updates every doc that change affects, in the same commit.**
A change isn't done until the docs say what the code and results now say. Before committing, go
through this table and update each row that applies:

| When a session… | Update |
|---|---|
| **Runs any experiment** (study, smoke run, first-pass or official run, gate or robustness test on EC2) | The study note `docs/studies/SN-*.md`: setup actually used, date, result table/chart, gap vs prediction. The **course**: that module's lab `status`, `measured` text with the headline numbers, instance and git tag (see "Course" below). README status table. Raw output stays in `results/` (git-ignored). |
| Builds or tags a milestone | README status table; the plan's checklist if scope changed; the study note's Setup and prediction prompts; the course module (`status: "ready"`, setup rows, command); the journal (`docs/journal/journal.html`, code tab via `scripts/journal/gen_code_docs.py`). |
| Makes or changes a design decision | `docs/decisions/README.md` row and the `NN-*.md` note; the plan's design-review row; the course's decision card (`D({...})`: status, choice, reason); the appendix decision log in `docs/course/src/05-unit3-5.js`. |
| Changes scope, order, hardware or models | The plan, README, and the course's units, modules and Orientation tables. |
| Adds or changes a script, flag or command | `benchmarks/README.md`, the study note's Setup command, and the course lab's setup rows. |
| Changes SLOs, gates or workloads | Decision 3/4/2 notes, the plan's testing program, `benchmarks/README.md`, the course Orientation tables. |

If a change makes any statement in these docs untrue, fix the statement, even outside the table.

## Course (learning dossier)
- Source of truth: `docs/course/src/` (data in `.js`, rendered by `06-app.js`). Build with
  `python docs/course/build.py` → `docs/course/index.html`. `--check` fails if the build is stale.
- Published artifact: https://claude.ai/artifact/FJmfEniZ8Pzc4o2BvU7552 . After every course change,
  rebuild, then republish `docs/course/index.html` with the Artifact tool, passing that URL as `url`
  (read it first in a new session, as the tool requires). The repo copy and the artifact must match.
- Recording a result in a module: set `measured:"<numbers>, <instance>, tag <tag>, <date>; see docs/studies/SN-*.md"`,
  and set `status` (`"ready"` = script exists, `"planned"`, `"optional"`). Keep the "Predict before you run"
  questions; never put numbers from a run into the prediction prompts.
- Every number in the course should match the study note it came from.

## Other rules
- Predict before measuring; correctness gates (G0–G2) pass before any study is measured.
- Compare against Hugging Face on the same device only.
- Commit messages name the milestone or decision (e.g. "M7: chunked prefill …", "Decision 10: …").
