# ekfdmd

`ekfdmd` is a small Python package for sequential and time-varying Dynamic Mode Decomposition, centered on Extended Kalman Filter DMD (EKF-DMD).

It complements [PyDMD](https://github.com/PyDMD/PyDMD): PyDMD provides a broad collection of DMD variants, while this package focuses on EKF-based and sequential operator estimation.

## Methods

- `EKFDMD`: EKF-DMD with a sparse augmented-state Jacobian.
- `EKFDMDc`: EKF-DMD with known control input.
- `SlidingWindowDMD`: DMD over overlapping time windows.
- `SlidingWindowDMDc`: sliding-window DMD with control.
- `OnlineDMD`: recursive least-squares online DMD with a forgetting factor.

## Installation

```sh
pip install ekfdmd
```

Numba is optional. When installed, it accelerates the Online DMD update loop without changing the public API.

```sh
pip install 'ekfdmd[numba]'
```

## Usage

```python
import numpy as np
from ekfdmd import EKFDMD

model = EKFDMD(
    state_noise=1e-3,
    measurement_noise=1e-2,
    operator_noise=0.0,
)
model.fit(X)  # X.shape == (n_features, n_snapshots)

A_t = model.operators
lambdas_t = model.eigs
modes_t = model.modes
x_next = model.predict(X[:, -1])
```

Controlled systems use the same `fit`-first interface:

```python
from ekfdmd import EKFDMDc

model = EKFDMDc().fit(X, U)
x_next = model.predict(X[:, -1], U[:, -1])
```

Time-varying modes can be reordered consistently with MAC-based tracking:

```python
from ekfdmd.postprocess import track_modes

tracked_eigs, tracked_modes = track_modes(model.eigs, model.modes)
```

`track_modes` uses the final snapshot as the reference ordering and matches adjacent mode matrices backward in time with the modal assurance criterion (MAC).

## Scope

This package intentionally does not include plotting, neural-network integrations, experiment management, or general data utilities. PyDMD remains the recommended package when EKF or sequential estimation is not required.

## License

MIT
