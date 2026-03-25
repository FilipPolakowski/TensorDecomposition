# CP Decomposition Pipeline

Cross-patient sEEG tensor decomposition — built on top of the supervisor's `sparsify.py`.

---

## What this does

Each patient has sEEG electrodes in different brain locations, making direct cross-patient comparison impossible. This pipeline:

1. Maps every patient's channels onto **1000 shared brain clusters** (MNI space) using the supervisor's clustering logic
2. Stacks all patients into a **3D tensor** — time × clusters × patients
3. Runs **CP decomposition** to find shared neural patterns across all patients

The result is three factor matrices describing *when* each pattern activates, *where* in the brain, and *which patients* show it.

---

## Files

| File | What it does |
|---|---|
| `cp_pipeline.py` | Full pipeline — sparsify, build tensor, normalise, decompose |
| `visualize_cp.py` | Plots the results — run after the pipeline |
| `diagnose.py` | Checks signal quality per patient (amplitude, NaN detection) |
| `HOW_TO_RUN.md` | Run commands |

---

## Quick start

```powershell
cd "C:\path\to\your\data"
python cp_pipeline.py --data_dir . --rank 7
```

---

## Pipeline steps

### 1. Sparsify
Each patient's electrode channels are matched to the supervisor's cluster IDs from `elecs.npz`.
A **binary mask** is built alongside the sparse matrix — marking which clusters have real data.
Without the mask, CP decomposition would try to fit the 94% of empty entries as if they were real signals.

### 2. Exclusion criteria
Patients are excluded if they fail any of these checks:

| Reason | Example |
|---|---|
| File not found | Patient in atlas but no `.npz` file |
| NaN or Inf in signal | Hardware or recording error |
| Wrong recording length | Must be exactly 6000 samples |
| No electrode matched any cluster | Channel naming mismatch in atlas |
| Known data quality issue | kh003 — hardware saturation at ~70 µV |

A full exclusion report is printed every run.

### 3. Normalisation
Robust scaling per patient per cluster — **median and IQR** instead of mean and standard deviation.
Reason: two patients (kh010, kh033) had unusually large amplitudes that would distort z-scoring for everyone else.
Values beyond ±10 IQR are clipped as likely recording artifacts.

### 4. CP decomposition
Approximates the tensor as a sum of R rank-1 components:

```
X  ≈  Σ  a_r ∘ b_r ∘ c_r     (r = 1 to R)
```

| Factor | Shape | Meaning |
|---|---|---|
| `A` | T × R | Temporal profile of each component |
| `B` | 1000 × R | Spatial brain map of each component |
| `C` | P × R | Patient loadings — who expresses each pattern |

Two challenges solved:
- **Missing data (94% empty)** — iterative imputation: fill blanks with current reconstruction, update, refill, repeat
- **Scale (6000 × 1000 × 88)** — randomised block sampling: use 30% of time steps per update

### 5. Output
Factor matrices saved to `cp_factors.npz` in your data folder.
Run `visualize_cp.py` to generate plots.

---

## Excluded patients (current dataset)

| Patient | Reason |
|---|---|
| kh001 | 12000 NaN values in signal |
| kh003 | Signal saturation — hardware clipping at ~70 µV |
| kh004 | 12000 NaN values in signal |
| kh009 | 12000 NaN values in signal |
| kh020 | Recording length 5155 (required 6000) |
| kh029 | Recording length 800 (required 6000) |
| kh046 | Recording length 5100 (required 6000) |
| kh060 | No electrode matched any cluster |
| kh062 | No electrode matched any cluster |
| kh073 | Recording length 3500 (required 6000) |

---

## References

| Method | Paper |
|---|---|
| CP decomposition | [Kolda & Bader (2009)](https://doi.org/10.1137/07070111X) — SIAM Review |
| Missing data | [Acar et al. (2011)](https://doi.org/10.1016/j.chemolab.2010.08.004) — Chemometrics |
| Randomised sampling | [Vervliet & De Lathauwer (2016)](https://doi.org/10.1109/JSTSP.2015.2503260) — IEEE |
| Rank selection | [Bro & Kiers (2003)](https://doi.org/10.1002/cem.801) — J. Chemometrics |
| Robust statistics | [Rousseeuw & Leroy (1987)](https://doi.org/10.1002/0471725382) — Wiley |
