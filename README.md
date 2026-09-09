# ekfdmd

`ekfdmd` is a small Python package for sequential and time-varying Dynamic Mode Decomposition, centered on Extended Kalman Filter DMD (EKF-DMD).

It complements [PyDMD](https://github.com/PyDMD/PyDMD): PyDMD provides a broad collection of DMD variants, while this package focuses on EKF-based and sequential operator estimation.

## Quickstart

See [`examples/nonomura_2019.ipynb`](examples/nonomura_2019.ipynb) for a reproducible implementation of the small-DoF numerical experiment in Section 4.1 of Nonomura, Shibata, and Takaki (2019).

## Methods

| Solver | Input | Main parameters | Purpose |
| --- | --- | --- | --- |
| `EKFDMD` | `X: (n, T)` | `state_noise`, `operator_noise`, `measurement_noise`, `rank` | EKF-based simultaneous state filtering and operator estimation |
| `CuPyEKFDMD` | `X: (n, T)` | same as `EKFDMD` | CUDA implementation of EKF-DMD for explicit GPU execution |
| `EKFDMDc` | `X: (n, T)`, `U: (m, T)` | same as `EKFDMD` | EKF-DMD with a known control input and estimated `A, B` |
| `SlidingWindowDMD` | `X: (n, T)` | `window_size`, `svd_rank` | independent DMD fits on overlapping windows |
| `SlidingWindowDMDc` | `X: (n, T)`, `U: (m, T)` | `window_size`, `svd_rank` | sliding-window DMD with control |
| `OnlineDMD` | `X: (n, T)` | `rho`, `initial_covariance` | recursive least-squares DMD with exponential forgetting |
| `RecursiveDMDc` | `X: (n, T)`, `U: (m, T)` | `rho`, `initial_covariance`, `rank` | recursive least-squares DMD with control and exponential forgetting |

## Installation

```sh
uv add ekfdmd
```

Numba is optional. When installed, it accelerates the Online DMD update loop without changing the public API.

```sh
uv add 'ekfdmd[numba]'
```

CUDA 12.x support is optional. The `cuda12` extra installs CuPy and the CUDA component wheels, so a system-wide CUDA Toolkit is not required when a compatible NVIDIA driver is already installed.

```sh
uv add 'ekfdmd[cuda12]'
```

For a source checkout, install the CUDA extra with:

```sh
uv sync --extra cuda12
```

## Usage

```python
from ekfdmd import EKFDMD

model = EKFDMD(
    state_noise=1e-3,
    measurement_noise=1e-2,
    operator_noise=0.0,
).fit(X)

A_t = model.operators
lambdas_t = model.eigs
modes_t = model.modes
x_next = model.predict(X[:, -1])
```

Use the CUDA backend explicitly when desired. Backend selection is not automatic because the CPU/GPU break-even point depends on the hardware and problem size. `CuPyEKFDMD` accepts NumPy input and returns NumPy arrays through the same data contract as `EKFDMD`.

```python
from ekfdmd.cupy import CuPyEKFDMD

model = CuPyEKFDMD(
    state_noise=1e-3,
    measurement_noise=1e-2,
    operator_noise=0.0,
).fit(X)
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
| `control_operators` | control-operator history | `(T, n, m)` for EKFDMDc/RecursiveDMDc, `(W, n, m)` for SlidingWindowDMDc |
| `eigs` | eigenvalue history | `(T, r)` or `(W, r)` when fixed-rank |
| `modes` | mode history | `(T, n, r)` or `(W, n, r)` when fixed-rank |
| `predicted_data` | causal one-step prior during `fit()` | `(n, T)`; EKFDMD/EKFDMDc/OnlineDMD/RecursiveDMDc only |
| `filtered_data` | EKF posterior state history | `(n, T)`; EKFDMD/EKFDMDc only |

`SlidingWindowDMD(svd_rank=None)` deliberately allows the numerical rank to vary by window, so `eigs` and `modes` are tuples of per-window arrays. Set `svd_rank` to obtain fixed-shape arrays suitable for mode tracking. `SlidingWindowDMDc` reports eigenpairs of each full state operator, so its modal dimension is always `n`.

When `EKFDMDc(initialize_with_dmd=False)` is used, the initial state operator is identity and the initial control operator is zero.

## Mode tracking

```python
from ekfdmd.postprocess import track_modes

tracked_eigs, tracked_modes = track_modes(model.eigs, model.modes)
```

`track_modes(eigs, modes)` expects fixed-shape histories `(steps, modes)` and `(steps, features, modes)`. The final snapshot defines the reference ordering, and adjacent mode matrices are matched backward in time by maximizing the modal assurance criterion (MAC).

## References

The implementation author and the authors of the underlying algorithms are distinct. Cite the software itself using `CITATION.cff`; cite the relevant method paper when reporting scientific results.

- T. Nonomura, H. Shibata, and R. Takaki, “Extended-Kalman-filter-based dynamic mode decomposition for simultaneous system identification and denoising,” *PLoS ONE*, 14(2), e0209836, 2019. https://doi.org/10.1371/journal.pone.0209836
- J. L. Proctor, S. L. Brunton, and J. N. Kutz, “Dynamic Mode Decomposition with Control,” *SIAM Journal on Applied Dynamical Systems*, 15(1), 142–161, 2016. https://doi.org/10.1137/15M1013857
