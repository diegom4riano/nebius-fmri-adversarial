# When Gradient Descent Fails: Second-Order Adversarial Attacks on Brain Imaging Models

*Submitted to the Nebius Serverless AI Builders Challenge — Healthcare & Life Sciences*

---

## The Clinical Stakes

Machine learning models are increasingly used in clinical neuroscience: predicting disease, staging conditions, and classifying brain states from fMRI. But how robust are they? An adversarial attacker — a clinician submitting slightly modified scans, or a model audit gone wrong — could flip a diagnosis with imperceptible perturbations.

To find out, I ran six adversarial attacks against **STAGIN** (Spatio-Temporal Attention Graph Isomorphism Network), a state-of-the-art model for sex classification from resting-state fMRI, on **216 Human Connectome Project subjects**. The result was surprising: a second-order Newton-CG attack achieved **60.7% attack success rate at ε=0.001** — while PGD with five times the compute budget achieved only **13.1%**. The gap traces to a single number: the Hessian condition number κ = **178,695**.

---

## The Model: STAGIN on HCP Data

STAGIN processes dynamic functional connectivity: it slides a 50-TR window across the 1,200-TR resting-state fMRI recording, computes a 333×333 ROI correlation matrix per window (~50 windows), and feeds the resulting spatio-temporal graph through 4 Graph Isomorphism Network layers with self-attention readout and a Transformer over the time axis.

Trained on the HCP Young Adult dataset, the model reaches **BACC = 77.2%** on the test split (216 subjects: 117 Female, 99 Male). This matches published benchmarks. The model checkpoint is 1.5 MB — small enough to run repeatedly inside a GPU job.

---

## The Adversarial Setup

The threat model is **L∞ perturbations on the FC matrix inputs** — perturbing correlation values in the sliding-window adjacency tensor. This is meaningful: an attacker with access to preprocessing could introduce subtle, bounded noise in the correlation estimation step.

I define **True Attack Success Rate (True ASR)** as the fraction of Male subjects (class 1, the non-target class) whose prediction flips to Female (class 0) after the attack. This is a targeted, subject-level metric that avoids inflated rates from trivially easy examples.

Six attacks ran across five epsilon values (0.001, 0.005, 0.01, 0.05, 0.1):

| Attack | Type | Budget |
|---|---|---|
| **Newton-CG** | 2nd order (Hessian) | 5 outer × 50 CG iters = 250 HVPs |
| **PGD-40** | 1st order | 40 gradient steps |
| **PGD-500** | 1st order (matched) | 500 gradient steps ≈ same FLOPs as Newton-CG |
| **AutoAttack** | Ensemble (APGD-CE + Square) | — |
| **APGD-CE** | 1st order w/ step schedule | 100 steps |
| **C&W L2** | 1st order, L2 norm | 50 steps |

---

## Why Condition Number Is the Hidden Variable

Before running the attacks, I estimated the Hessian condition number κ via Rayleigh quotients on random directions: **κ = 178,695**. This means the loss surface has directions that are 178,000× steeper than others.

For gradient descent (PGD), this is catastrophic. Each step moves freely in the flat directions and barely at all in the steep ones — or overshoots and oscillates. With ε=0.001 (a very small budget), the gradient iterates are trapped: PGD takes 500 steps and achieves 13% ASR.

Newton-CG solves the system `H·d = -g` at each outer step, finding the Newton direction that accounts for curvature. Even with only 5 outer steps, it navigates the ill-conditioned landscape efficiently — reaching 60.7% ASR in less total wall-clock time than PGD-500.

---

## Results Across the Epsilon Sweep

The table below shows True ASR (% of Male subjects flipped to Female) for all attacks across five epsilon values:

| Attack | ε=0.001 | ε=0.005 | ε=0.01 | ε=0.05 | ε=0.1 |
|---|---|---|---|---|---|
| **Newton-CG** | **60.7%** | **58.5%** | **50.0%** | **92.7%** | **93.9%** |
| APGD-CE | 22.6% | 30.5% | 29.3% | 86.6% | 74.4% |
| PGD-40 | 31.0% | 35.4% | 41.5% | 54.9% | 53.7% |
| AutoAttack | 17.9% | 15.9% | 28.1% | 62.2% | 65.9% |
| C&W L2 | 16.7% | 18.3% | 18.3% | 18.3% | 18.3% |
| PGD-500 | 13.1% | 29.3% | 42.7% | 51.2% | 53.7% |

Full results: `output/attack_results.json` in the repository.

Three patterns stand out:

**1. Newton-CG dominates at tight budgets.** At ε=0.001, Newton-CG achieves **60.7% ASR in 741s** while PGD-500 achieves only **13.1% in 3,771s** — a 4.6× gap. PGD-500 is even worse than PGD-40, a direct consequence of ill-conditioning: more steps with a tiny step size amplify oscillations rather than making progress.

**2. The gap closes as epsilon grows — then reopens.** By ε=0.01, PGD-500 recovers to 42.7% and nearly matches Newton-CG (50%). But at ε=0.05–0.1, Newton-CG jumps to 93%+ while PGD plateaus around 53%. The large epsilon ball is not enough to save PGD when the Newton direction is available.

**3. C&W L2 is epsilon-agnostic.** C&W L2 stays at ~18% across all epsilon values. This is expected: C&W minimizes L2 perturbation, and its effective budget in L∞ coordinates doesn't grow in the same way. The model's vulnerability is aligned with L∞ directions, not L2.

The **H200 peak VRAM was 86.9 GB** — exceeding the 80 GB A100 limit. The H200 SXM (141 GB HBM3e) was the minimum viable GPU for this experiment.

---

## The Infrastructure: Nebius Serverless AI Jobs on H200

Running six attacks with double-backward Hessian-vector products on 216 subjects requires serious GPU memory. The STAGIN forward pass with batch=32 peaks at **86.9 GB VRAM** — beyond what an A100 (80 GB) can fit. The **Nebius H200 SXM** (141 GB HBM3e) handled this comfortably.

The entire compute infrastructure runs serverless via **Nebius AI Jobs**:

```bash
nebius ai job create \
  --platform gpu-h200-sxm \
  --preset 1gpu-16vcpu-200gb \
  --volume storagebucket-...:/workspace/data \
  --container-command bash \
  --args '-c "pip install -r requirements.txt && python test_fmri_model.py ..."'
```

No persistent VM, no cluster management. The job provisions an H200 node, mounts the S3 bucket at `/workspace/data`, runs the attack script, writes results back to S3, and releases the hardware. Total cost: well within the $100 challenge budget.

Code, model checkpoint, and results are at: **[github.com/diegom4riano/nebius-fmri-adversarial](https://github.com/diegom4riano/nebius-fmri-adversarial)**

---

## Conclusion

Second-order attacks exploit information that first-order methods ignore: the curvature of the loss landscape. For clinical neural networks operating on high-dimensional brain connectivity data, this matters — the Hessian can be extremely ill-conditioned (κ≈180k here), making gradient descent ineffective even with a generous compute budget.

The practical implication: **robustness evaluations using only PGD may significantly underestimate a model's true vulnerability.** A Newton-CG attack with the same budget reveals weaknesses that PGD simply cannot find.

All code, configs, and full results are open-source. The Nebius H200 serverless infrastructure made this experiment reproducible in a single `make deploy-attack` command — no cluster setup, no persistent billing.

---

*#NebiusServerlessChallenge — Healthcare & Life Sciences*

*Repository: [github.com/diegom4riano/nebius-fmri-adversarial](https://github.com/diegom4riano/nebius-fmri-adversarial)*
