# Neuro-Fuzzy Deception Detection Using Keystroke Hesitation Patterns

> Course project: Neuro-Fuzzy Applications (interdisciplinary team of 4)
> Status: Foundation / planning stage

---

## 1. One-Paragraph Summary

We build a system that estimates how likely a typed answer is to be **fabricated** rather than **truthful**, using only *how* the person types (key hold times, gaps between keys, pauses before typing) and not *what* they type. Raw keystroke timings are converted into fuzzy linguistic variables (e.g. "short / normal / long pause"), and an **ANFIS** (Adaptive Neuro-Fuzzy Inference System) learns and tunes the fuzzy rules from data to output a **deception score between 0 and 1**. The idea rests on the cognitive-load theory of lying: constructing a false answer is mentally harder than recalling a true one, and that can show up as hesitation.

**Honest scope statement (include this in the report):** The training data is **semi-synthetic**. A small real pilot sample is collected from the team, and a larger dataset is generated from its measured statistics. Results show that the *pipeline works on data with an assumed/observed effect*. They do **not** prove that hesitation reliably detects lying in the real world.

---

## 2. Goals and Non-Goals

**Goals**
- Build a working end-to-end pipeline: keystroke timings -> features -> fuzzy sets -> ANFIS -> deception score.
- Show that the neuro-fuzzy model works and is **interpretable** (we can print and explain the learned rules).
- Compare ANFIS against at least two baselines (a fixed-threshold rule and an SVM or logistic regression).
- Deliver a simple live demo: type an answer, get a score.

**Non-Goals**
- Claiming a real-world lie detector.
- Collecting a large human-subjects dataset.
- Building a polished production app.

---

## 3. System Pipeline

```
 [1] Data Source            [2] Feature Extraction        [3] Fuzzification
 pilot logger + synthetic -> dwell, flight, pause,     ->  membership functions
 generator (labelled)        digraph, backspace, etc.      (short / normal / long)
                                                                  |
                                                                  v
 [6] Demo / UI          <-  [5] Deception Score       <-  [4] ANFIS
 type -> score              threshold + confidence        trains & tunes rules
                                   |
                                   v
                          [7] Evaluation vs baselines
```

### Stage details

| Stage | Input | Output | Notes |
|---|---|---|---|
| 1. Data source | Typing sessions | Rows of raw key events (key, down-time, up-time, label) | Real pilot + synthetic expansion |
| 2. Feature extraction | Raw key events | Numeric feature vector per answer | See feature list below |
| 3. Fuzzification | Feature values | Membership degrees per fuzzy set | Initial Gaussian/bell MFs, later tuned by ANFIS |
| 4. ANFIS | Fuzzy features | Continuous output 0..1 | Takagi-Sugeno rules, hybrid learning |
| 5. Scoring | ANFIS output | Deception score + label | Threshold chosen on validation set |
| 6. Demo | Live typing | Score on screen | Simple CLI or small web page |
| 7. Evaluation | Test predictions | Metrics, plots | Accuracy, precision, recall, F1, ROC-AUC |

### Candidate features (keep to 4-6 for ANFIS)

ANFIS rule count grows exponentially with inputs (`MFs ^ inputs`), so we **select a small set**. Candidates:

1. `mean_dwell`: average key hold time
2. `mean_flight`: average gap between key-up and next key-down
3. `initial_latency`: time from question shown to first keystroke
4. `pause_count`: number of pauses longer than a threshold (e.g. 1s)
5. `backspace_ratio`: backspaces / total keys (self-correction)
6. `typing_speed_var`: variance of inter-key intervals

Start with 4 inputs x 3 MFs = 81 rules (manageable). Pick the final set using feature importance or correlation analysis on the pilot data.

---

## 4. Repository Structure

```
neuro-fuzzy-deception/
├── README.md                  # Short overview + how to run
├── PROJECT_FOUNDATION.md      # This file
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                   # Real pilot keystroke logs (CSV)
│   ├── synthetic/             # Generated datasets (CSV)
│   └── processed/             # Feature tables ready for ANFIS
│
├── src/
│   ├── collection/
│   │   ├── logger.py          # Captures keydown/keyup timestamps
│   │   └── prompts.py         # Truthful vs fabricate question set
│   ├── synthetic/
│   │   └── generate.py        # Seeded generator with deception effect
│   ├── features/
│   │   └── extract.py         # Raw events -> feature vector
│   ├── model/
│   │   ├── anfis_model.py     # Wrapper around chosen ANFIS library
│   │   ├── train.py           # Training loop + config
│   │   └── baselines.py       # Threshold rule, SVM / logistic regression
│   ├── scoring/
│   │   └── score.py           # ANFIS output -> score + decision
│   └── demo/
│       └── app.py             # Live demo (CLI or small web UI)
│
├── notebooks/
│   ├── 01_pilot_analysis.ipynb      # What does real data look like?
│   ├── 02_synthetic_validation.ipynb# Does synthetic match pilot stats?
│   ├── 03_anfis_training.ipynb
│   └── 04_evaluation.ipynb
│
├── results/
│   ├── figures/               # MF plots, ROC curves, confusion matrices
│   ├── rules/                 # Exported learned fuzzy rules (text)
│   └── metrics.json
│
├── docs/
│   ├── report.md              # Final report draft
│   └── slides/                # Presentation
│
└── tests/
    └── test_features.py       # Basic sanity tests for extraction
```

---

## 5. Data Plan

### 5.1 Real pilot data (small, essential)
- **Who:** the 4 team members, plus friends if easy (aim for 8-10 people).
- **Protocol:** each person answers ~20 prompts typed into the logger:
  - ~10 **truthful** prompts (e.g. "Describe what you had for breakfast today.")
  - ~10 **fabricate** prompts (e.g. "Describe a breakfast you did NOT have, convincingly, as if true.")
- **Control:** randomise prompt order, match prompt length/difficulty across conditions, same keyboard where possible.
- **Label:** known by design (we instruct which condition applies).
- **Record per answer:** every key event with down/up timestamps, plus the time the prompt appeared.

### 5.2 Synthetic expansion
- Compute per-condition statistics from the pilot (mean/std of each feature for truthful vs fabricated).
- Generate N (e.g. 2,000-5,000) samples by sampling from fitted distributions (log-normal is a good fit for timing data), with **per-user baseline offsets** so different "users" type at different speeds.
- Make the deception effect size **a parameter** so we can run experiments such as "what happens if the effect is small vs large?"
- Always seed the random generator for reproducibility.

### 5.3 Rules for honesty
- Label all synthetic files clearly (`synthetic_` prefix).
- Report pilot sample size and the measured effect, even if it is small or absent.
- Evaluate on a **subject-held-out split** (test users not seen in training) to avoid inflated scores.

---

## 6. Model Plan

### 6.1 ANFIS choice
Candidate libraries (check current state and licence before depending on them):
- `twmeggs/anfis`: Python ANFIS implementation, simple to read.
- `sanfis` (PyTorch): more modern, easier to integrate with standard training code.

**Decision rule:** try `sanfis` first for PyTorch flexibility; fall back to `twmeggs/anfis` if setup is painful. Whichever we choose, we write our own thin wrapper (`anfis_model.py`) so the rest of the code does not care.

### 6.2 Configuration to explore
- Number of inputs: 3 to 5
- MFs per input: 2 or 3 (Gaussian)
- Epochs, learning rate
- Output: regression to 0..1 then thresholded, or binary classification

### 6.3 Interpretability deliverable
Export the learned rules in readable form, e.g.:
> IF initial_latency is *long* AND pause_count is *high* THEN deception score is *high*

and plot membership functions **before vs after** training. This is the strongest argument for "why neuro-fuzzy and not just SVM."

---

## 7. Evaluation Plan

| Item | Detail |
|---|---|
| Split | Train / validation / test, split **by subject** |
| Metrics | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix |
| Baseline 1 | Fixed-threshold rule on one feature (e.g. initial latency) |
| Baseline 2 | SVM or logistic regression on same features |
| Ablations | ANFIS with fewer/more MFs; feature subsets |
| Sensitivity | Vary synthetic effect size; show where the model stops working |
| Repeats | 5 random seeds, report mean and std |

**Success criterion (realistic):** ANFIS is competitive with baselines **and** yields human-readable rules. It does not need to beat every baseline.

---

## 8. Team Roles (4 people)

| Role | Owns | Main deliverables |
|---|---|---|
| **A: Data & Collection** | Logger, prompt set, pilot sessions, synthetic generator | `logger.py`, `prompts.py`, `generate.py`, datasets |
| **B: Features & Fuzzification** | Feature extraction, feature selection, initial MFs | `extract.py`, pilot analysis notebook |
| **C: Neuro-Fuzzy Model** | ANFIS wrapper, training, tuning, rule export | `anfis_model.py`, `train.py`, rules, MF plots |
| **D: Evaluation, Demo & Report** | Baselines, metrics, demo, report, slides | `baselines.py`, `app.py`, `report.md`, slides |

Domain-branch members (psychology / HCI / signal processing background, if any) should lead the **prompt design** and the **cognitive-load justification** in the report, since that is where their expertise matters most.

---

## 9. Timeline (relative; adjust once the end-term schedule is announced)

| Phase | Tasks | Target |
|---|---|---|
| **Week 1** | Finalise team + title; create repo; write logger; design prompts; install ANFIS library and run its example | Repo + logger working |
| **Week 2** | Run pilot sessions; extract features; analyse real effect; build synthetic generator | `data/raw` + `data/synthetic` |
| **Week 3** | Train ANFIS; tune MFs/rules; build baselines | First results |
| **Week 4** | Full evaluation, sensitivity tests, rule export; build demo | Results + demo |
| **Week 5** | Report, slides, rehearsal, buffer | Submission-ready |

**Critical path:** logger -> pilot data -> synthetic generator -> features -> ANFIS. Nothing downstream can start until the generator exists, so Role A should prioritise it.

---

## 10. Tech Stack

- **Language:** Python 3.10+
- **Core:** numpy, pandas, scikit-learn, matplotlib
- **Neuro-fuzzy:** `sanfis` + PyTorch (or `twmeggs/anfis`)
- **Keystroke capture:** `pynput` (desktop) or a small browser page using JS `keydown`/`keyup` events (easier across OSes and gives precise timestamps)
- **Demo:** Streamlit or a simple Flask page
- **Version control:** Git + GitHub (private until submission)

---

## 11. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Real pilot shows **no** hesitation effect | Medium | Report it honestly; use synthetic effect-size sweep to show what the pipeline does *if* an effect exists |
| ANFIS rule explosion / slow training | Medium | Limit to 4-5 inputs, 2-3 MFs each |
| ANFIS library is outdated or buggy | Medium | Wrapper layer; keep second library as fallback |
| Evaluator challenges "hesitation = lying" premise | High | Frame as cognitive-load hypothesis; cite limitations; position as pipeline demonstration |
| Overfitting to synthetic patterns | High | Subject-held-out split; validate synthetic stats against pilot |
| Time runs short | Medium | MVP first (3 features, 1 baseline), add extras only after |
| Ethics concerns | Low-Medium | Pilot participants consent; no personal data stored; no real "lie-detector" claims |

---

## 12. Ethics and Limitations (put in the report)

- Participants know they are in an experiment and are *instructed* to lie, so this measures "constructing a false answer on request," not real-world deceit.
- Typing behaviour varies widely across people, keyboards, and moods; a single-feature "hesitation means lying" claim is not scientifically established.
- The system must **not** be presented as a tool for judging real people.

---

## 13. Definition of Done

- [ ] Logger captures keydown/keyup timestamps correctly
- [ ] Pilot dataset collected and documented
- [ ] Synthetic generator reproduces pilot statistics (validated)
- [ ] Feature extraction tested
- [ ] ANFIS trained, learned rules exported, MF before/after plots saved
- [ ] Two baselines implemented and compared
- [ ] Evaluation with subject-held-out split and multiple seeds
- [ ] Working live demo
- [ ] Report with honest data disclosure and limitations
- [ ] Slides and rehearsed 5-minute pitch

---

## 14. First Three Actions

1. Create the GitHub repo with the folder structure above.
2. Pick the ANFIS library and get its **example** running on toy data.
3. Write the keystroke logger and record your own first 10 answers.
