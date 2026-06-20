"""Single source of truth for the pass@k inconsistency band.

The band is the pass@k range that counts as inconsistently solved: above it the model is
reliably right, below it reliably wrong, and inside it the same problem flips between correct
and wrong across resamples. Every harvester, the sweep, the reclassifier, and the aggregator
import BAND and in_band from here, so the bracket is defined in exactly one place. Widening or
narrowing it is the single edit below; nothing else needs to change.

At k=16 the current band of 0.125 to 0.875 spans 2 through 14 correct out of 16.
"""

BAND = (0.125, 0.875)


def in_band(pass_at_k, band=BAND):
    """True if a pass@k rate lands inside the inconsistency band."""
    low, high = band
    return low <= pass_at_k <= high
