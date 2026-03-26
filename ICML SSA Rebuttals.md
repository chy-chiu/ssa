---
categories:
  - "[[Categories/Notes|Notes]]"
tags:
  - note
created:
  - 2026-03-24
---
# ICML 2026 Rebuttal Plan

## Quick motivation / reality check
You are **not cooked**. This paper has a real idea, a strong topic, a 5 and a 4 already, and at least one reviewer who clearly sees the contribution. The current problem is **not that the work is empty**; it is that the paper is **under-defended** and somewhat **over-framed** for the amount of evidence currently shown.

So the goal over the next **96 hours** is **not** to “perfect the paper.”  
The goal is to:

1. **stabilize the 5 and 4**
2. **flip at least one 2 upward**
3. **show intellectual honesty + added evidence**
4. **reframe claims tightly so the AC can justify accepting a promising but stylized paper**

And if this still misses ICML, that does **not** mean the project is bad. It would still be very alive for **NeurIPS / ICLR / FAccT / AIES / EC / ACM EAAMO / Management Science-ish workshop pipelines**, depending on how you sharpen the angle. But for now: **head down, 96 hours, maximum leverage only.**

---

## Core rebuttal thesis
We should consistently push this framing across all replies:

- [ ] **AI-Work is a mechanism-isolation testbed**, not a predictive simulator of real AI labor markets
- [ ] The main claims are **qualitative and directional**, not quantitative forecasts
- [ ] The paper studies **one plausible class of AI labor market mechanisms**
- [ ] The SSA result is **not** “we invented a deep new architecture”; it is:
  - [ ] a **minimal structured prompting intervention**
  - [ ] showing that **explicit elicitation of metacognition, competitive awareness, and planning improves economic outcomes**
- [ ] The trace-scoring analysis is **descriptive / diagnostic only**
- [ ] The intervention + ablation are the main evidence for the capability story

---

# Priority 0 — Non-negotiables (must finish first)

## To write
- [ ] Draft a **master rebuttal framing paragraph** used across all reviewer responses:
  - [ ] thank reviewers
  - [ ] acknowledge stylization
  - [ ] explicitly narrow the claim to mechanism-isolation / qualitative insights
  - [ ] say we added clarifications + robustness checks
- [ ] Draft a **2–3 sentence honest concession**:
  - [ ] “We agree the environment is stylized”
  - [ ] “Our goal is not predictive realism but controlled study of adverse selection, reputation, and investment under competition”
  - [ ] “We have revised the framing accordingly”
- [ ] Prepare **one compact list of concrete changes** to cite in each reviewer response:
  - [ ] clearer framing
  - [ ] fixed appendix refs
  - [ ] moved key equations to main text
  - [ ] added human gig-market context
  - [ ] added new robustness experiments / ablations

## Experiments
- [ ] Decide **today** which new experiments are feasible in 96 hours
- [ ] Prioritize only experiments that can move reviewer scores:
  - [ ] robustness to assumptions
  - [ ] SSA control
  - [ ] scale
  - [ ] no-benchmark / reduced-benchmark reputation update
- [ ] Make one final call on what is **too expensive** to finish in time and drop it

---

# Priority 1 — Highest leverage reviewer: GrSA (score 2, conf 5)

## Goal
- [ ] Move GrSA from **2 → 3 or 4**
- [ ] This is the most important review to address substantively

## Main concerns to answer
- [ ] Equilibrium may be **model-induced**
- [ ] Market sizes are **too small**
- [ ] SSA gains may be **prompt length / information injection**

## To write
- [ ] Write a response that **starts with new evidence**, not explanation
- [ ] First sentence should explicitly say:
  - [ ] “We thank the reviewer; we agree robustness and scale are central, and we have now added/ran X, Y, Z”
- [ ] Add a **tight claim-narrowing sentence**:
  - [ ] “Our intent is not to claim these are universal AI labor-market equilibria, but that these mechanisms induce consistent qualitative behaviors in a controlled testbed”
- [ ] Add a **defense of stylized assumptions**:
  - [ ] Cobb-Douglas / multiplicative scoring approximates reputation-price tradeoffs
  - [ ] Gale–Shapley provides a tractable client-optimal matching rule
  - [ ] reputation update is a reduced-form informational mechanism
  - [ ] training is a reduced-form human-capital / capability investment mechanism
- [ ] Add a **strong sentence on SSA**:
  - [ ] “We agree SSA is a structured prompt intervention; our claim is not architectural novelty, but that consistently eliciting these reasoning patterns improves outcomes within the same backbone model”
- [ ] Add a **sentence explicitly referencing new controls** if run:
  - [ ] length-matched / information-matched control
  - [ ] alternative scoring rules
  - [ ] reduced benchmark updating
  - [ ] larger N
- [ ] End with:
  - [ ] “We hope these additions address the reviewer’s concerns regarding robustness and relevance”

## Experiments
- [ ] **Robustness to client scoring function**
  - [ ] linear score instead of Cobb-Douglas
  - [ ] CES variant if cheap
  - [ ] compare whether qualitative findings still hold:
    - [ ] open bidding lowers prices
    - [ ] performance pay increases training
    - [ ] SSA still outperforms controls
- [ ] **Robustness to reputation dynamics**
  - [ ] remove or weaken benchmark-based reputation recalibration when not hired
  - [ ] alternative update window / forgetting factor
  - [ ] test whether main qualitative results persist
- [ ] **Scale experiment**
  - [ ] run N = 32 at minimum
  - [ ] run N = 64 if feasible
  - [ ] report whether qualitative trends persist
- [ ] **SSA length / information control**
  - [ ] construct a prompt with similar length / detail but without explicit M/C/P scaffold
  - [ ] or an information-matched baseline with generic strategic advice
  - [ ] compare against SSA on same backbone

---

# Priority 2 — Second movable reject: 5M91 (score 2, conf 3)

## Goal
- [ ] Move 5M91 from **2 → 3**
- [ ] This reviewer seems more movable via clarity + positioning

## Main concerns to answer
- [ ] Too much is in appendix
- [ ] framework feels artificial / only two actions
- [ ] capability section feels qualitative
- [ ] LLM-as-judge correlations feel weak
- [ ] overall clarity poor

## To write
- [ ] Open by directly acknowledging clarity problems
  - [ ] “We agree the initial submission relied too heavily on the appendix and contained unresolved placeholders”
- [ ] State clearly what was changed in the paper:
  - [ ] key equations moved to main text
  - [ ] appendix refs fixed
  - [ ] model names clarified
  - [ ] better motivation for the three capability domains
- [ ] Defend the **two-action abstraction**:
  - [ ] not meant as full realism
  - [ ] chosen to isolate a core explore/exploit tradeoff
  - [ ] complexity comes from sequential strategy under competition and partial information, not action count alone
- [ ] Clarify the **three capability domains**
  - [ ] motivated from the economics of hidden quality + strategic competition
  - [ ] metacognition = self-calibration under noisy signals
  - [ ] competitive awareness = inference over rival behavior and market niches
  - [ ] planning = dynamic skill investment vs immediate revenue
- [ ] Downgrade the rhetorical weight of the **trace-scoring correlation**
  - [ ] explicitly say it is descriptive only
  - [ ] say the main evidence is intervention + component ablation
- [ ] Close by emphasizing:
  - [ ] the contribution is not realism-maximal simulation
  - [ ] it is a tractable benchmark/testbed for studying agent economic behavior

## Experiments
- [ ] If possible, include **one robustness result** in this reply too
- [ ] If possible, include **one non-LLM / stronger outcome-based validation angle**
  - [ ] e.g. component ablation is outcome-based and does not depend on LLM-as-judge
- [ ] If feasible, rerun a **stochastic training** version to counter “too artificial”

---

# Priority 3 — Protect the 4: t6s6

## Goal
- [ ] Keep at **4**, maybe nudge sentiment upward

## Main concerns to answer
- [ ] realism is limited
- [ ] paper presentation needs revision
- [ ] Appendix ?? placeholders
- [ ] section pointers and abbreviations unclear
- [ ] wants brief intro to human gig platforms

## To write
- [ ] Thank reviewer for recognizing the modeling tradeoff
- [ ] Explicitly say we **agree realism is limited** and have revised the framing to emphasize:
  - [ ] controlled testbed
  - [ ] qualitative mechanism study
  - [ ] not full-market simulation
- [ ] List paper fixes clearly:
  - [ ] fixed all Appendix ?? placeholders
  - [ ] added appendix section on human gig-economy platforms
  - [ ] added explicit appendix pointers in Section 3.1
  - [ ] expanded model names in main text
  - [ ] moved key formulas into main text
- [ ] Mention any robustness checks you added
- [ ] Keep the tone simple and grateful; this reviewer is not the one to fight with

## Experiments
- [ ] None required specifically for this reviewer beyond the shared robustness work
- [ ] If available, mention additional validation / robustness concisely

---

# Priority 4 — Protect the 5: TW1h

## Goal
- [ ] Keep at **5**
- [ ] Show engagement and seriousness

## Main concerns to answer
- [ ] metacognition measure vs prompt-visible signals
- [ ] possible structural bias in LLM-as-judge
- [ ] deterministic upskilling is stylized

## To write
- [ ] Thank them for the thoughtful and constructive review
- [ ] Clarify metacognition carefully:
  - [ ] the agent does see partial public signals (reputation, outcomes)
  - [ ] metacognition here refers to **inferring latent competitiveness from noisy public signals**
  - [ ] not mere reading of prompt variables
- [ ] Explicitly say:
  - [ ] “We agree the trace-scoring is descriptive; the intervention and ablation are the stronger evidence”
- [ ] Address LLM-as-judge bias:
  - [ ] mention we checked / added bias analysis if possible
  - [ ] otherwise acknowledge limitation and add to revised discussion
- [ ] Respond positively to the stochastic upskilling suggestion
  - [ ] say added if you run it
  - [ ] say good future direction if you do not

## Experiments
- [ ] **Stochastic upskilling ablation**
  - [ ] training success probabilistic rather than deterministic
  - [ ] show SSA / market-design conclusions still directionally hold if possible
- [ ] **LLM-as-judge bias check** if feasible
  - [ ] prompt-order swap / anonymization
  - [ ] compare whether SSA wording alone inflates judge scores

---

# Priority 5 — Shared paper changes reviewers may notice at a glance

## To write
- [ ] Revise abstract / intro / discussion language to reduce overclaiming
- [ ] Replace “future AI labor markets will behave like X” style wording with:
  - [ ] “AI-Work provides a tractable environment for studying how such mechanisms may shape agent behavior”
- [ ] Add one sentence explicitly noting:
  - [ ] “Some conclusions are mechanism-contingent and should be interpreted as directional”
- [ ] Move the following into main text if not already:
  - [ ] scoring formula
  - [ ] reputation update summary
  - [ ] training update summary
  - [ ] concise SSA description
- [ ] Fix all typography / grammar / appendix issues
- [ ] Add brief human gig-economy platform overview in appendix

## Experiments
- [ ] None beyond shared experiments above

---

# Priority 6 — Specific experiments ranked by ROI

## Tier A — Highest ROI
- [ ] **Alternative scoring function robustness**
- [ ] **Reduced/no benchmark recalibration reputation variant**
- [ ] **SSA prompt control (length/information matched)**
- [ ] **N = 32 scale run**

## Tier B — Strong if feasible
- [ ] **N = 64 scale run**
- [ ] **stochastic upskilling**
- [ ] **alternative reputation forgetting/window**
- [ ] **client-side scoring sensitivity sweep**

## Tier C — Nice but only if time remains
- [ ] **LLM-as-judge bias stress test**
- [ ] **additional contract-cost variant**
- [ ] **more elaborate macro validation**

---

# Reviewer-by-reviewer actual reply drafting checklist

## To write
- [ ] Draft **GrSA** response first
- [ ] Draft **5M91** response second
- [ ] Draft **t6s6** response third
- [ ] Draft **TW1h** response last

## Per-review reply structure
- [ ] 1 sentence thank you
- [ ] 1 sentence acknowledging the core concern
- [ ] 2–4 sentences on what was changed / added
- [ ] 1 sentence clarifying the revised scope of claims
- [ ] 1 sentence closing respectfully

## Style constraints
- [ ] Do **not** sound defensive
- [ ] Do **not** overstate what your new experiments prove
- [ ] Do **not** argue that reviewers are wrong about stylization
- [ ] Do **not** claim realism you do not have
- [ ] Do explicitly claim:
  - [ ] tractability
  - [ ] mechanism isolation
  - [ ] qualitative robustness
  - [ ] same-backbone intervention evidence for SSA

---

# 96-hour execution plan

## First 12 hours
### To write
- [ ] Lock the new framing language
- [ ] Draft master rebuttal paragraph
- [ ] Outline reviewer-specific replies
- [ ] Make exact list of experiments that will be run

### Experiments
- [ ] Launch highest-priority runs:
  - [ ] alt scoring
  - [ ] no/reduced benchmark recalibration
  - [ ] N = 32
  - [ ] SSA control

## Hours 12–36
### To write
- [ ] Draft full GrSA response
- [ ] Draft full 5M91 response
- [ ] Start text revisions in paper for consistency

### Experiments
- [ ] Monitor runs
- [ ] Start stochastic upskilling if compute allows
- [ ] Start N = 64 only if N = 32 is stable

## Hours 36–60
### To write
- [ ] Draft t6s6 response
- [ ] Draft TW1h response
- [ ] Insert exact experiment results into replies
- [ ] Tighten wording and remove any overclaiming

### Experiments
- [ ] Finish remaining runs
- [ ] Generate compact summary table / plots

## Hours 60–84
### To write
- [ ] Finalize all rebuttals
- [ ] Finalize key paper edits reviewers might notice
- [ ] Cross-check consistency between rebuttal and revised manuscript

### Experiments
- [ ] Re-run anything broken / incomplete
- [ ] Verify statistical summaries / numbers

## Final 12 hours
### To write
- [ ] Ruthless trim for clarity
- [ ] Check tone: confident, honest, non-defensive
- [ ] Proofread everything
- [ ] Submit

### Experiments
- [ ] None unless absolutely necessary

---

# Final mental reminders
- [ ] I do **not** need to prove this is the definitive model of AI labor markets
- [ ] I only need to show this is a **useful, principled, and partially robust mechanism testbed**
- [ ] I do **not** need to win every argument
- [ ] I **do** need to give the AC a believable reason to accept despite stylization
- [ ] The most important thing is to **move one or both reject reviewers upward**
- [ ] This paper still has a path
- [ ] 96 hours of focused work can absolutely change the outcome









天動說 example
single agent diverse start