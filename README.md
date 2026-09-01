# VarStar Lab

Turns a night's worth of astronomical exposures into a light curve, measures the
star's pulsation period, and converts that period into a distance.

Built for the project proposal **“Photographing and Modeling Variable Stars”**
(Luke Prasarttongosoth, Aidan Hawkins, Craig Collar), whose target is **V0756 CrA**.

Runs at **http://localhost:12113**

```bash
cd ~/varstar_lab
python3 app.py          # then open http://localhost:12113
```

---

## Why a period gives you a distance

V0756 CrA is classified **HADS(B)** — a high-amplitude δ Scuti, double-mode. That
classification is the whole reason a distance is reachable. δ Scuti stars obey a
**period–luminosity relation**: how fast the star pulses tells you how intrinsically
bright it is. Compare that with how bright it *looks*, and the difference is distance.
Same logic Henrietta Leavitt used on Cepheids, one rung down the ladder.

```
1.  aperture photometry on every frame        →  flux(t) for target + comparisons
2.  Δm = −2.5 log₁₀(F_target / ΣF_comp)       →  light curve, cloud and airmass divided out
3.  Lomb–Scargle + phase dispersion min.      →  pulsation period P
4.  M_V = −2.94 log₁₀P − 1.34                 →  absolute magnitude   (Ziaali+ 2019)
5.  d = 10^((⟨V⟩ − M_V − A_V)/5 + 1) pc       →  distance
```

### Target reference (from the AAVSO VSX record in the proposal)

| | |
|---|---|
| Name | V0756 CrA (ASAS J182536-4213.6, TYC 7909-809-1) |
| Position | 18 25 36.26 −42 13 35.8 (J2000) = 276.40108° −42.22661° |
| Galactic | l = 352.088°, b = −13.434° |
| Type | HADS(B) — high-amplitude δ Scuti, double-mode |
| V range | 11.43 – 12.0 |
| Catalog period | 0.1071934 d = **2.57264 h** |
| Epoch | HJD 2453600.038 |

---

## The two things that decide whether this works

**1. A distance needs an *apparent* magnitude, not a differential one.**
A differential light curve has no absolute scale. Enter catalog V magnitudes for at
least one comparison star (AAVSO's Variable Star Plotter, or APASS) and the app
derives a zero point; or type the mean apparent V directly. Without one of those,
step 5 refuses and tells you why.

**2. One continuous run must cover at least one full pulsation cycle.**
This is the finding that most shapes the observing plan. A sampling experiment on
this star's period:

| observing pattern | cycles per session | period recovered |
|---|---|---|
| 3 nights × 1.5 h | 0.6 | **wrong by 9.5%** — locks onto the +1 c/d alias |
| 4 nights × 1.5 h | 0.6 | still wrong |
| 6 nights × 1.5 h | 0.6 | still wrong |
| 3 nights × 2.0 h | 0.8 | correct |
| 3 nights × 3.0 h | 1.2 | correct, +0.06% |

Observing from one site on a 24-hour planet imprints a 1 cycle/day pattern on the
sampling, so every real frequency *f* is shadowed by peaks at *f* ± n c/d. **Adding
more short nights raises every alias together and never resolves the ambiguity** —
only covering a full cycle inside one uninterrupted session does, because then the
shape of the curve itself constrains the frequency.

The proposal's planned “continuous images 2–3 hrs” sits right at the 2.57 h
threshold. **3+ hours continuous, on each of 2–3 nights, is the version that
yields a period you can defend.** The app measures this and says so.

---

## What it does with your data

**Ingest** — FITS (`.fits/.fit/.fts`), TIFF, PNG/JPEG, and camera RAW
(`.cr2/.cr3/.nef/.arw/.dng`, read linearly with no white balance or gamma).
Mid-exposure timestamps recovered from `DATE-OBS`/`MJD-OBS`/`BJD` or EXIF, with
`+EXPTIME/2` applied; if there is no timestamp anywhere, you supply a start time
and cadence. **FITS is strongly preferred** — 8-bit files have already been
stretched for display, which flattens the star cores photometry depends on, and
the app will tell you when that has happened.

**Photometry** — DAOStarFinder detection; FWHM from the radial profile at half
maximum (not second moments, which Moffat wings inflate by ~2×); every frame
re-registered by phase cross-correlation plus per-star centroiding, because the
proposal turns auto-guiding off and the field walks tens of pixels; sky from the
sigma-clipped median of an annulus; errors from the CCD equation. Saturated stars
and frames are excluded and reported, never silently included.

**Differential reduction** — target ÷ comparison ensemble. A per-star *check
scatter* (each comparison against the others) exposes a comparison star that is
itself variable — the classic way a spurious period enters a light curve.

**Period** — Lomb–Scargle *and* phase dispersion minimisation, which assume
different things and fail differently; their disagreement is used as an error
estimate. Iterative prewhitening searches for the second radial mode that HADS(**B**)
implies, and checks the period ratio against the 0.756–0.787 fundamental/first-overtone
band. Competing aliases are listed as **candidates you can click to lock onto**,
rather than hidden behind one number.

Quoted period uncertainty is the **largest** of: the analytic formula (Montgomery &
O'Donoghue 1999), a residual bootstrap that preserves the observing window, the
Lomb-Scargle-vs-PDM disagreement, and — below 1.5 cycles of coverage — the
frequency-resolution limit P²/T. The analytic value alone assumes white noise, one
coherent sinusoid, and that the right peak was picked; on a short run it understates
the real error by up to a factor of 75. Measured against injected truth:

| coverage | true period error | quoted σ | verdict |
|---|---|---|---|
| 1.25 h, 1 session | 52% | 99% | Insufficient |
| 2.5 h, 1 session | 3.5% | 100% | Insufficient |
| 3 × 1.5 h (alias trap) | 9.5% | 11% | Insufficient |
| 4.8 h, 1 session | 2.3% | 1.0% | Provisional |
| 3 × 3.0 h | 0.06% | 0.02% | Provisional |

Note the third column never badly understates the second — that is the property the
error bar exists to have, and it is what the app is tuned for. Also note that
“cycles covered” is computed from the period *found*: if that period is wrong, the
coverage looks better than it is, which is precisely why the sub-1.5-cycle regime is
treated as unmeasured rather than merely imprecise.

**Distance** — three published δ Scuti P–L relations (Ziaali et al. 2019 default;
McNamara 2011; Nemec, Nemec & Lutz 1994 for metal-poor SX Phe) plus custom
coefficients. Extinction as A_V = 3.1 E(B−V), fetchable from the IRSA/SFD dust map.
Full error propagation with an itemised budget. Optional T_eff unlocks luminosity,
radius, and a mass estimate from the pulsation constant.

**Independent check** — one button queries the **Gaia DR3 parallax** for the same
star and reports the agreement in sigma. This is the best possible validation of
the whole chain.

**Export** — light curve CSV (one row per exposure), a `report.json` with every
number and the method notes, and a ZIP with print-ready white-background figures.

---

## Verified end to end

Synthetic FITS frames are generated with realistic defects — field drift, seeing
and transparency variations, a cloud event, Poisson and read noise, hot pixels —
and a genuine double-mode HADS curve injected. The pipeline is given no knowledge
of the injected values.

Three nights × 3 h (648 frames), reduced through the HTTP API exactly as the
browser does:

| quantity | injected | recovered |
|---|---|---|
| period | 0.1071934 d | 0.1072537 d — **+0.056%** |
| peak-to-peak amplitude | 0.570 mag | 0.574 mag |
| mean apparent V | 11.715 | 11.704 ± 0.020 |
| overtone period ratio | 0.774 | 0.776 |
| per-point precision | — | 1.84 mmag |

Distance from that period: **980 pc (3196 ly) ± 7.0%**.
Gaia DR3 parallax for V0756 CrA: 0.9341 ± 0.0294 mas → **1071 pc (3492 ly)**.
Agreement: **1.18σ**. The error budget is dominated by the P–L relation's intrinsic
scatter (0.146 of 0.152 mag), not by the photometry — which is the honest conclusion:
better data will not shrink this error much, because the relation itself is the limit.

```bash
python3 selftest.py three_nights   # pipeline in-process, prints truth vs recovered
python3 apitest.py three_nights    # full HTTP API, all plots and exports
python3 uploadtest.py              # real multipart uploads, 8-bit files, alias trap,
                                   # and the refusal paths
```

---

## Simulation presets

| preset | what it is for |
|---|---|
| `single_night` | 2.5 h, ~1 cycle — what the proposal currently plans |
| `short_night` | 1.25 h — deliberately inadequate, worth seeing once |
| `alias_trap` | 3 × 1.5 h — FAP of 1e-183 and a 9% wrong answer |
| `three_nights` | 3 × 3.0 h — a period you can quote |
| `clean` | 4.8 h guided and photometric — the pipeline's noise floor |

---

## Layout

```
app.py                 FastAPI server, session state, background jobs
pipeline/
  ingest.py            file reading, channel selection, timestamps
  photometry.py        detection, drift tracking, apertures, differential reduction
  timing.py            HJD / BJD light-travel-time corrections, target constants
  period.py            Lomb-Scargle, PDM, Fourier fits, aliases, uncertainties
  distance.py          P-L relations, extinction, distance modulus, Gaia lookup
  synth.py             synthetic observing runs
  plotting.py          matplotlib figures (dark for screen, light for print)
static/                single-page UI
selftest.py apitest.py uploadtest.py
```

Config via environment: `VARSTAR_PORT` (12113), `VARSTAR_MAX_FRAMES` (1200),
`VARSTAR_MAX_UPLOAD_MB` (4096). Sessions live in memory with files under `data/`,
expiring after 6 hours.

## References

- Ziaali, Bedding, Murphy, Bland-Hawthorn & Hey (2019), MNRAS 486, 4348 — δ Scuti P–L
- McNamara (2011), AJ 142, 110 — δ Scuti / SX Phe P–L
- Nemec, Nemec & Lutz (1994), AJ 108, 222 — SX Phoenicis P–L
- Montgomery & O'Donoghue (1999), DSSN 13, 28 — frequency uncertainty
- Stellingwerf (1978), ApJ 224, 953 — phase dispersion minimisation
- Schlafly & Finkbeiner (2011) — extinction recalibration
- VanderPlas (2018), ApJS 236, 16 — Lomb–Scargle in practice
