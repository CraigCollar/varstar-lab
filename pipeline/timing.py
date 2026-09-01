"""Time-system corrections.

Light travel time across Earth's orbit is up to +-8.3 minutes. Over a single
2-3 h run the correction is nearly constant, so it barely touches the measured
period - but it matters as soon as nights are combined, and it is required for
any time of maximum that will be compared against a published epoch.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

# V0756 CrA, from the AAVSO VSX record quoted in the project proposal.
TARGET_DEFAULT = {
    "name": "V0756 CrA",
    "ra_str": "18 25 36.26",
    "dec_str": "-42 13 35.8",
    "ra_deg": 276.40108,
    "dec_deg": -42.22661,
    "gal_l": 352.088,
    "gal_b": -13.434,
    "vtype": "HADS(B)",
    "mag_min": 12.0,
    "mag_max": 11.43,
    "period_cat": 0.1071934,
    "epoch_cat": 2453600.038,
    "other_names": "ASAS J182536-4213.6, ASASSN-V J182536.32-421335.7, TYC 7909-809-1",
}


def parse_coord(ra: str, dec: str) -> SkyCoord:
    """Parse coordinates given either as decimal degrees or as sexagesimal.

    The unit for RA has to be decided *before* handing the string to SkyCoord,
    not by trying degrees and falling back. "18 25 36.26" parses perfectly well
    as 18d25m36.26s = 18.43 deg, so a try/except would silently accept a
    position 258 degrees away from the intended one.

    Rule: a bare decimal number is degrees; anything sexagesimal (spaces,
    colons, or h/m/s markers) means RA is in hours.
    """
    ra, dec = str(ra).strip(), str(dec).strip()
    if not ra or not dec:
        raise ValueError("both RA and Dec are required")

    def is_plain_decimal(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    if is_plain_decimal(ra) and is_plain_decimal(dec):
        return SkyCoord(float(ra) * u.deg, float(dec) * u.deg, frame="icrs")

    # Sexagesimal: hours for RA, degrees for Dec (the universal convention).
    coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame="icrs")
    return coord


def convert_times(jd_utc, ra=None, dec=None, site: Optional[dict] = None,
                  system: str = "bjd_tdb"):
    """Convert JD(UTC) to HJD(UTC) or BJD(TDB).

    Returns (converted array, label, mean correction in seconds). Falls back to
    the input times, clearly labelled, if the coordinates or site are missing.
    """
    jd_utc = np.asarray(jd_utc, dtype=float)
    system = (system or "none").lower()
    if system in ("none", "jd", "jd_utc") or ra is None or dec is None:
        return jd_utc, "JD (UTC, uncorrected)", 0.0

    ok = np.isfinite(jd_utc)
    if not ok.any():
        return jd_utc, "JD (UTC, uncorrected)", 0.0

    try:
        coord = parse_coord(ra, dec)
        loc = None
        if site and site.get("lat") is not None and site.get("lon") is not None:
            loc = EarthLocation.from_geodetic(
                lon=float(site["lon"]) * u.deg,
                lat=float(site["lat"]) * u.deg,
                height=float(site.get("elev") or 0.0) * u.m,
            )
        t = Time(jd_utc[ok], format="jd", scale="utc", location=loc)
        if system.startswith("bjd"):
            ltt = t.light_travel_time(coord, kind="barycentric")
            corrected = (t.tdb + ltt).jd
            label = "BJD (TDB)" + ("" if loc is not None else ", geocentric")
        else:
            ltt = t.light_travel_time(coord, kind="heliocentric")
            corrected = (t.utc + ltt).jd
            label = "HJD (UTC)" + ("" if loc is not None else ", geocentric")
        out = jd_utc.copy()
        out[ok] = corrected
        delta = float(np.mean((corrected - jd_utc[ok]) * 86400.0))
        return out, label, delta
    except Exception as exc:  # pragma: no cover - depends on IERS data
        return jd_utc, f"JD (UTC, uncorrected - conversion failed: {exc})", 0.0
