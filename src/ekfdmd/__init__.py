"""Sequential Dynamic Mode Decomposition methods centered on EKF-DMD."""

from ._dmd import SlidingWindowDMD, SlidingWindowDMDc
from ._ekf import EKFDMD, EKFDMDc
from ._online import OnlineDMD, RecursiveDMDc

__all__ = [
    "EKFDMD",
    "EKFDMDc",
    "OnlineDMD",
    "RecursiveDMDc",
    "SlidingWindowDMD",
    "SlidingWindowDMDc",
]
