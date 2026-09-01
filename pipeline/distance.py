"""Turning a period into a distance.

Delta Scuti stars - including the high-amplitude subclass HADS, which is what the
proposal's target V0756 CrA is - obey a period-luminosity relation. Measure the
pulsation period, read off the absolute magnitude, compare with the observed mean
apparent magnitude, and the distance modulus gives the distance. This is the same
Leavitt-law logic used for Cepheids, on a fainter and shorter-period rung of the
same ladder.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

R_V = 3.1                 # standard total-to-selective extinction ratio
PC_PER_LY = 3.261563777
T_SUN = 5772.0            # K
MBOL_SUN = 4.74           # absolute bolometric magnitude of the Sun
Q_FUNDAMENTAL = 0.033     # days, radial fundamental pulsation constant

# M_V = slope * log10(P / day) + intercept
PL_RELATIONS = {
    "ziaali2019": {
        "label": "Ziaali et al. (2019) - Gaia DR2 delta Scuti",
        "slope": -2.94, "slope_err": 0.06,
        "intercept": -1.34, "intercept_err": 0.06,
        "scatter": 0.12,
        "band": "V",
        "note": ("Calibrated on Gaia DR2 parallaxes for 1276 delta Scuti stars. The "
                 "current default for field delta Scuti work."),
        "valid": (0.02, 0.25),
    },
    "mcnamara2011": {
        "label": "McNamara (2011) - delta Scuti / SX Phe",
        "slope": -2.89, "slope_err": 0.13,
        "intercept": -1.31, "intercept_err": 0.10,
        "scatter": 0.15,
        "band": "V",
        "note": ("Pre-Gaia calibration from cluster and field stars. Agrees with "
                 "Ziaali within errors, which is a useful consistency check."),
        "valid": (0.02, 0.30),
    },
    "nemec1994": {
        "label": "Nemec, Nemec & Lutz (1994) - SX Phoenicis",
        "slope": -3.725, "slope_err": 0.30,
        "intercept": -1.933, "intercept_err": 0.30,
        "scatter": 0.20,
        "band": "V",
        "note": ("Steeper relation derived for metal-poor SX Phe pulsators. Use only "
                 "if the star is known to be metal-poor / Population II; it will give "
                 "a systematically different distance for a normal-metallicity HADS."),
        "valid": (0.02, 0.20),
    },
}


def absolute_magnitude(period, sigma_period=0.0, relation="ziaali2019",
                       custom: Optional[dict] = None):
    """M_V from the period, with full error propagation.

    sigma_MV^2 = (slope * sigma_logP)^2 + (logP * sigma_slope)^2
                 + sigma_intercept^2 + sigma_intrinsic^2
    """
    if custom:
        rel = {
            "label": custom.get("label", "custom relation"),
            "slope": float(custom["slope"]),
            "slope_err": float(custom.get("slope_err", 0.0)),
            "intercept": float(custom["intercept"]),
            "intercept_err": float(custom.get("intercept_err", 0.0)),
            "scatter": float(custom.get("scatter", 0.0)),
            "band": custom.get("band", "V"),
            "note": "User-supplied coefficients.",
            "valid": None,
        }
        key = "custom"
    else:
        key = relation if relation in PL_RELATIONS else "ziaali2019"
        rel = PL_RELATIONS[key]

    period = float(period)
    if period <= 0:
        raise ValueError("period must be positive")
    log_p = math.log10(period)

    m_v = rel["slope"] * log_p + rel["intercept"]

    sigma_period = float(sigma_period or 0.0)
    sigma_logp = sigma_period / (period * math.log(10)) if sigma_period > 0 else 0.0

    var = (rel["slope"] * sigma_logp) ** 2
    var += (log_p * rel["slope_err"]) ** 2
    var += rel["intercept_err"] ** 2
    var += rel["scatter"] ** 2
    sigma_mv = math.sqrt(var)

    warn = None
    if rel.get("valid"):
        lo, hi = rel["valid"]
        if not (lo <= period <= hi):
            warn = (f"Period {period:.5f} d falls outside the relation's calibrated "
                    f"range {lo}-{hi} d. The extrapolation may be unreliable.")

    return {
        "relation_key": key,
        "relation_label": rel["label"],
        "relation_note": rel.get("note"),
        "band": rel["band"],
        "slope": rel["slope"], "intercept": rel["intercept"],
        "log_period": log_p,
        "sigma_log_period": sigma_logp,
        "M": m_v, "sigma_M": sigma_mv,
        "terms": {
            "from_period": abs(rel["slope"] * sigma_logp),
            "from_slope": abs(log_p * rel["slope_err"]),
            "from_intercept": rel["intercept_err"],
            "intrinsic_scatter": rel["scatter"],
        },
        "warning": warn,
    }


def extinction(ebv=None, a_v=None, sigma=None):
    """A_V from either E(B-V) (times R_V = 3.1) or a direct A_V."""
    if a_v is not None and a_v != "":
        a = float(a_v)
        sig = float(sigma) if sigma not in (None, "") else 0.3 * a
        return {"a_v": a, "sigma_a_v": sig, "ebv": a / R_V, "source": "A_V entered directly"}
    if ebv is not None and ebv != "":
        e = float(ebv)
        a = R_V * e
        sig = float(sigma) * R_V if sigma not in (None, "") else max(0.03, 0.16 * a)
        return {"a_v": a, "sigma_a_v": sig, "ebv": e,
                "source": f"A_V = 3.1 x E(B-V) = 3.1 x {e:.3f}"}
    return {"a_v": 0.0, "sigma_a_v": 0.0, "ebv": 0.0,
            "source": "no extinction applied (distance will be an underestimate)"}


def distance_from_modulus(mean_mag, sigma_mag, abs_mag, sigma_abs_mag,
                          a_v=0.0, sigma_a_v=0.0):
    """Distance modulus -> distance, with an asymmetric confidence interval.

    mu_0 = <m> - M - A     and     d = 10^(mu_0/5 + 1) pc

    The interval is computed by pushing mu_0 +- sigma through the exponential,
    which is why the upper and lower bars differ.
    """
    mu_app = float(mean_mag) - float(abs_mag)
    mu_0 = mu_app - float(a_v)
    var = float(sigma_mag) ** 2 + float(sigma_abs_mag) ** 2 + float(sigma_a_v) ** 2
    sigma_mu = math.sqrt(var)

    d = 10.0 ** (mu_0 / 5.0 + 1.0)
    d_lo = 10.0 ** ((mu_0 - sigma_mu) / 5.0 + 1.0)
    d_hi = 10.0 ** ((mu_0 + sigma_mu) / 5.0 + 1.0)
    sigma_d_sym = d * math.log(10) / 5.0 * sigma_mu

    return {
        "mu_apparent": mu_app,
        "mu_0": mu_0,
        "sigma_mu": sigma_mu,
        "a_v": float(a_v), "sigma_a_v": float(sigma_a_v),
        "distance_pc": d,
        "distance_pc_lo": d_lo,
        "distance_pc_hi": d_hi,
        "sigma_pc": sigma_d_sym,
        "distance_ly": d * PC_PER_LY,
        "distance_ly_lo": d_lo * PC_PER_LY,
        "distance_ly_hi": d_hi * PC_PER_LY,
        "sigma_ly": sigma_d_sym * PC_PER_LY,
        "distance_kpc": d / 1000.0,
        "parallax_mas": 1000.0 / d if d > 0 else float("nan"),
        "error_budget": {
            "mean_magnitude": float(sigma_mag),
            "absolute_magnitude": float(sigma_abs_mag),
            "extinction": float(sigma_a_v),
            "total_modulus": sigma_mu,
            "relative_distance_pct": 100.0 * sigma_d_sym / d if d > 0 else float("nan"),
        },
    }


def solve(period, sigma_period, mean_mag, sigma_mag, relation="ziaali2019",
          ebv=None, a_v=None, sigma_ext=None, custom=None):
    """One-call period -> distance."""
    mv = absolute_magnitude(period, sigma_period, relation, custom)
    ext = extinction(ebv, a_v, sigma_ext)
    dist = distance_from_modulus(mean_mag, sigma_mag, mv["M"], mv["sigma_M"],
                                 ext["a_v"], ext["sigma_a_v"])
    return {"absolute": mv, "extinction": ext, "distance": dist,
            "period": float(period), "sigma_period": float(sigma_period or 0.0),
            "mean_mag": float(mean_mag), "sigma_mean_mag": float(sigma_mag)}


# --------------------------------------------------------------------------
# derived stellar properties (bonus - answers the proposal's mass question)
# --------------------------------------------------------------------------

def stellar_properties(abs_mag, sigma_abs_mag, period, teff=None, bc_v=0.0,
                       sigma_bc=0.10):
    """Luminosity, and - given a temperature - radius, density and mass.

    Delta Scuti stars are A-F type, where the V-band bolometric correction is
    within ~0.03 mag of zero, so the luminosity is well determined from M_V alone.
    Radius needs a temperature (the proposal gets one from spectroscopy); mass then
    follows from the pulsation constant Q = P sqrt(rho/rho_sun), which for the
    radial fundamental mode is ~0.033 d. Treat the mass as order-of-magnitude.
    """
    m_bol = float(abs_mag) + float(bc_v)
    sigma_mbol = math.sqrt(float(sigma_abs_mag) ** 2 + float(sigma_bc) ** 2)
    lum = 10.0 ** (0.4 * (MBOL_SUN - m_bol))
    sigma_lum = lum * 0.4 * math.log(10) * sigma_mbol

    out = {
        "m_bol": m_bol, "bc_v": float(bc_v),
        "luminosity_lsun": lum, "sigma_luminosity_lsun": sigma_lum,
        "radius_rsun": None, "sigma_radius_rsun": None,
        "density_rho_sun": None, "mass_msun": None, "teff": None,
        "q_assumed": Q_FUNDAMENTAL,
        "caveat": ("Luminosity assumes BC_V = %.2f +- %.2f mag, safe for an A-F "
                   "pulsator. Mass uses the fundamental-mode pulsation constant "
                   "Q = %.3f d and is an estimate, not a measurement."
                   % (bc_v, sigma_bc, Q_FUNDAMENTAL)),
    }

    if teff:
        teff = float(teff)
        if teff <= 0:
            return out
        radius = math.sqrt(lum) * (T_SUN / teff) ** 2
        # dR/R = 0.5 dL/L (temperature error not propagated - user-supplied)
        sigma_r = radius * 0.5 * (sigma_lum / lum) if lum > 0 else None
        rho = (Q_FUNDAMENTAL / float(period)) ** 2      # in solar mean densities
        mass = rho * radius ** 3
        out.update({
            "teff": teff,
            "radius_rsun": radius, "sigma_radius_rsun": sigma_r,
            "density_rho_sun": rho, "mass_msun": mass,
        })
    return out


# --------------------------------------------------------------------------
# external cross-checks (optional, need network access)
# --------------------------------------------------------------------------

def fetch_ebv(ra_deg: float, dec_deg: float, timeout: int = 30):
    """Line-of-sight reddening from the IRSA dust service (SFD / Schlafly)."""
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astroquery.ipac.irsa.irsa_dust import IrsaDust

        coord = SkyCoord(float(ra_deg) * u.deg, float(dec_deg) * u.deg, frame="icrs")
        tbl = IrsaDust.get_query_table(coord, section="ebv", timeout=timeout)
        row = tbl[0]
        cand = [c for c in tbl.colnames if "SandF" in c and "mean" in c.lower()]
        if cand:
            ebv = float(row[cand[0]])
            ref = "Schlafly & Finkbeiner (2011) recalibration"
        else:
            cand = [c for c in tbl.colnames if "SFD" in c and "mean" in c.lower()]
            if not cand:
                return {"ok": False, "error": "unexpected IRSA response columns"}
            ebv = float(row[cand[0]])
            ref = "Schlegel, Finkbeiner & Davis (1998)"
        return {
            "ok": True, "ebv": ebv, "a_v": R_V * ebv, "reference": ref,
            "note": ("This is the *total* reddening through the Galaxy along this "
                     "sight line. A star closer than the full dust column is reddened "
                     "less, so treat it as an upper limit."),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def fetch_gaia(ra_deg: float, dec_deg: float, radius_arcsec: float = 5.0,
               timeout: int = 60):
    """Gaia DR3 parallax for the nearest source - an independent distance check."""
    try:
        from astroquery.gaia import Gaia

        Gaia.ROW_LIMIT = 5
        adql = f"""
        SELECT TOP 5 source_id, ra, dec, parallax, parallax_error,
               phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
               ruwe, pmra, pmdec,
               DISTANCE(POINT({float(ra_deg)}, {float(dec_deg)}),
                        POINT(ra, dec)) AS sep
        FROM gaiadr3.gaia_source
        WHERE 1 = CONTAINS(POINT({float(ra_deg)}, {float(dec_deg)}),
                           CIRCLE(ra, dec, {float(radius_arcsec) / 3600.0}))
        ORDER BY sep ASC
        """
        job = Gaia.launch_job(adql)
        tbl = job.get_results()
        if len(tbl) == 0:
            return {"ok": False, "error": "no Gaia DR3 source within the search radius"}
        r = tbl[0]
        plx = float(r["parallax"])
        eplx = float(r["parallax_error"])
        out = {
            "ok": True,
            "source_id": str(r["source_id"]),
            "separation_arcsec": float(r["sep"]) * 3600.0,
            "parallax_mas": plx, "parallax_error_mas": eplx,
            "g_mag": float(r["phot_g_mean_mag"]),
            "ruwe": float(r["ruwe"]) if r["ruwe"] is not None else None,
        }
        if plx > 0:
            out["distance_pc"] = 1000.0 / plx
            out["distance_ly"] = 1000.0 / plx * PC_PER_LY
            out["distance_pc_lo"] = 1000.0 / (plx + eplx)
            out["distance_pc_hi"] = 1000.0 / (plx - eplx) if plx > eplx else float("inf")
            out["snr"] = plx / eplx if eplx > 0 else float("nan")
            out["note"] = ("Naive 1/parallax inversion. Valid here because the "
                           "parallax signal-to-noise is high; for faint or distant "
                           "stars a Bayesian distance is required instead.")
        else:
            out["note"] = "Parallax is negative - unusable for a distance."
        if out.get("ruwe") and out["ruwe"] > 1.4:
            out["warning"] = (f"RUWE = {out['ruwe']:.2f} > 1.4: the Gaia astrometric fit "
                              "is poor (possible binary), so this parallax is suspect.")
        return out
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def compare(measured: dict, gaia: dict):
    """Sigma-level agreement between the P-L distance and the Gaia parallax distance."""
    if not gaia.get("ok") or "distance_pc" not in gaia:
        return None
    d_m = measured["distance_pc"]
    s_m = measured["sigma_pc"]
    d_g = gaia["distance_pc"]
    s_g = abs(1000.0 / gaia["parallax_mas"] ** 2) * gaia["parallax_error_mas"]
    diff = d_m - d_g
    denom = math.sqrt(s_m ** 2 + s_g ** 2)
    nsig = abs(diff) / denom if denom > 0 else float("nan")
    if nsig < 1:
        verdict = "Excellent agreement (within 1 sigma)."
    elif nsig < 2:
        verdict = "Consistent (within 2 sigma)."
    elif nsig < 3:
        verdict = "Marginal tension (2-3 sigma)."
    else:
        verdict = ("Significant disagreement (>3 sigma). Check the photometric zero "
                   "point, the extinction, and whether the period is the radial "
                   "fundamental rather than an overtone or an alias.")
    return {
        "distance_pl_pc": d_m, "sigma_pl_pc": s_m,
        "distance_gaia_pc": d_g, "sigma_gaia_pc": s_g,
        "difference_pc": diff,
        "percent_difference": 100.0 * diff / d_g if d_g else float("nan"),
        "n_sigma": nsig, "verdict": verdict,
    }
