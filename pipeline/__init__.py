"""VarStar Lab reduction pipeline.

ingest      - read FITS / camera images, recover mid-exposure timestamps
photometry  - source detection, drift tracking, aperture + differential photometry
timing      - HJD / BJD time-system corrections
period       - Lomb-Scargle, PDM, Fourier fits, prewhitening, uncertainties
distance    - period-luminosity relations, extinction, distance modulus
synth       - synthetic observing runs for testing
plotting    - matplotlib figures as PNG bytes
"""

__all__ = ["ingest", "photometry", "timing", "period", "distance", "synth", "plotting"]
__version__ = "1.0.0"
