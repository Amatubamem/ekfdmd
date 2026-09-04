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
from ekfdmd import EKFDMD

model = EKFDMD(
    state_noise=1e-3,
    measurement_noise=1e-2,
    operator_noise=0.0,
).fit(X)  # X.shape == (n_features, n_snapshots)

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

`predict()` always means one-step prediction with the final fitted operator. It does not perform reconstruction or filtering.

## Public data contract

Let `n` be the full state dimension, `r` the reduced/modal dimension, `m` the control dimension, `T` the number of snapshots, and `W` the number of sliding windows.

| Attribute | Meaning | Shape |
| --- | --- | --- |
| `operator` | final state operator | `(n, n)` |
| `operators` | operator history | `(T, n, n)` for EKF/Online, `(W, n, n)` for sliding-window methods |
| `control_operator` | final control operator | `(n, m)` |
| `control_operators` | control-operator history | `(T, n, m)` for EKFDMDc, `(W, n, m)` for SlidingWindowDMDc |
| `eigs` | eigenvalue history | `(T, r)` or `(W, r)` when fixed-rank |
| `modes` | mode history | `(T, n, r)` or `(W, n, r)` when fixed-rank |
| `predicted_data` | causal one-step prior during `fit()` | `(n, T)`; EKFDMD/EKFDMDc/OnlineDMD only |
| `filtered_data` | EKF posterior state history | `(n, T)`; EKFDMD/EKFDMDc only |

`SlidingWindowDMD(svd_rank=None)` deliberately allows the numerical rank to vary by window, so `eigs` and `modes` are tuples of per-window arrays. Set `svd_rank` to obtain fixed-shape arrays suitable for mode tracking. `SlidingWindowDMDc` reports eigenpairs of each full state operator, so its modal dimension is always `n`.

When `EKFDMDc(initialize_with_dmd=False)` is used, the initial state operator is identity and the initial control operator is zero.

## Mode tracking

Time-varying modes can be reordered consistently with MAC-based tracking:

```python
from ekfdmd.postprocess import track_modes

tracked_eigs, tracked_modes = track_modes(model.eigs, model.modes)
```

`track_modes(eigs, modes)` expects fixed-shape histories `(steps, modes)` and `(steps, features, modes)`. The final snapshot defines the reference ordering, and adjacent mode matrices are matched backward in time by maximizing the modal assurance criterion (MAC).

## Scope

This package intentionally does not include plotting, neural-network integrations, experiment management, or general data utilities. PyDMD remains the recommended package when EKF or sequential estimation is not required.

## License

MIT
