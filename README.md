# ekfdmd

`ekfdmd` implements sequential and time-varying Dynamic Mode Decomposition methods centered on Extended Kalman Filter DMD (EKF-DMD).

For a broader collection of DMD variants, see [PyDMD](https://github.com/PyDMD/PyDMD).

## Quickstart

[`examples/nonomura_2019.ipynb`](examples/nonomura_2019.ipynb) reproduces the small-DoF numerical experiment from Section 4.1 of Nonomura, Shibata, and Takaki (2019).

## Methods

| Solver | Input | Main parameters | Purpose |
| --- | --- | --- | --- |
| `EKFDMD` | `X: (n, T)` | `state_noise`, `operator_noise`, `measurement_noise`, `rank` | simultaneous state filtering and operator estimation with EKF-DMD |
| `CuPyEKFDMD` | `X: (n, T)` | same as `EKFDMD` | CUDA implementation of EKF-DMD |
| `MPSEKFDMD` | `X: (n, T)` | same as `EKFDMD` | float32 Apple MPS implementation of EKF-DMD |
| `EKFDMDc` | `X: (n, T)`, `U: (m, T)` | same as `EKFDMD` | EKF-DMD with a known control input and estimated `A, B` |
| `SlidingWindowDMD` | `X: (n, T)` | `window_size`, `svd_rank` | independent DMD fits on overlapping windows |
| `SlidingWindowDMDc` | `X: (n, T)`, `U: (m, T)` | `window_size`, `svd_rank` | sliding-window DMD with control |
| `OnlineDMD` | `X: (n, T)` | `rho`, `initial_covariance` | recursive least-squares DMD with exponential forgetting |

## Installation

```sh
uv add ekfdmd
```

Numba accelerates the Online DMD update loop:

```sh
uv add 'ekfdmd[numba]'
```

For CUDA 12.x, install the CuPy backend and CUDA component wheels:

```sh
uv add 'ekfdmd[cuda12]'
```

A compatible NVIDIA driver is required. A system-wide CUDA Toolkit is not required.

For Apple MPS, install the PyTorch backend:

```sh
uv add 'ekfdmd[mps]'
```

For a source checkout, use `uv sync --extra cuda12` or `uv sync --extra mps`.

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

CUDA 12.x:

```python
from ekfdmd.cupy import CuPyEKFDMD

model = CuPyEKFDMD(
    state_noise=1e-3,
    measurement_noise=1e-2,
    operator_noise=0.0,
).fit(X)
```

`CuPyEKFDMD` accepts NumPy arrays and returns NumPy arrays with the same public attributes as `EKFDMD`.

Apple MPS:

```python
from ekfdmd.mps import MPSEKFDMD

model = MPSEKFDMD(
    state_noise=1e-3,
    measurement_noise=1e-2,
    operator_noise=0.0,
).fit(X)
```

`MPSEKFDMD` performs the EKF update in `float32`. Rounding differences from the default `float64` implementation can accumulate over long sequences. [`benchmarks/benchmark_mps_ekfdmd.py`](benchmarks/benchmark_mps_ekfdmd.py) measures runtime and numerical differences on the target Mac.

Controlled systems use the same `fit`-first interface:

```python
from ekfdmd import EKFDMDc

model = EKFDMDc().fit(X, U)
x_next = model.predict(X[:, -1], U[:, -1])
```

`predict()` returns a one-step prediction with the final fitted operator.

## Public data contract

Let `n` be the full state dimension, `r` the reduced/modal dimension, `m` the control dimension, `T` the number of snapshots, and `W` the number of sliding windows.

| Attribute | Meaning | Shape |
| --- | --- | --- |
| `operator` | final state operator | `(n, n)` |
| `operators` | operator history | `(T, n, n)` for EKF-DMD and OnlineDMD methods, `(W, n, n)` for sliding-window methods |
| `control_operator` | final control operator | `(n, m)` |
| `control_operators` | control-operator history | `(T, n, m)` for `EKFDMDc`, `(W, n, m)` for `SlidingWindowDMDc` |
| `eigs` | eigenvalue history | `(T, r)` or `(W, r)` when fixed-rank |
| `modes` | mode history | `(T, n, r)` or `(W, n, r)` when fixed-rank |
| `predicted_data` | causal one-step prior during `fit()` | `(n, T)` for EKF-DMD and OnlineDMD methods |
| `filtered_data` | EKF posterior state history | `(n, T)` for EKF-DMD implementations |

When `SlidingWindowDMD(svd_rank=None)` is used, the numerical rank may vary between windows. `eigs` and `modes` are then tuples of per-window arrays. Set `svd_rank` for fixed-shape arrays suitable for mode tracking. `SlidingWindowDMDc` reports eigenpairs of each full state operator, so its modal dimension is `n`.

When `EKFDMDc(initialize_with_dmd=False)` is used, the initial state operator is identity and the initial control operator is zero.

## Mode tracking

```python
from ekfdmd.postprocess import track_modes

tracked_eigs, tracked_modes = track_modes(model.eigs, model.modes)
```

`track_modes(eigs, modes)` expects fixed-shape histories `(steps, modes)` and `(steps, features, modes)`. The final snapshot defines the reference ordering, and adjacent mode matrices are matched backward in time by maximizing the modal assurance criterion (MAC).

## References

Use `CITATION.cff` to cite the software. Cite the relevant method paper when reporting scientific results.

- T. Nonomura, H. Shibata, and R. Takaki, “Extended-Kalman-filter-based dynamic mode decomposition for simultaneous system identification and denoising,” *PLoS ONE*, 14(2), e0209836, 2019. https://doi.org/10.1371/journal.pone.0209836
- J. L. Proctor, S. L. Brunton, and J. N. Kutz, “Dynamic Mode Decomposition with Control,” *SIAM Journal on Applied Dynamical Systems*, 15(1), 142–161, 2016. https://doi.org/10.1137/15M1013857
- H. Zhang, C. W. Rowley, E. A. Deem, and L. N. Cattafesta, “Online Dynamic Mode Decomposition for Time-Varying Systems,” *SIAM Journal on Applied Dynamical Systems*, 18(3), 1586–1609, 2019. https://doi.org/10.1137/18M1192329
- P. J. Schmid, “Dynamic mode decomposition of numerical and experimental data,” *Journal of Fluid Mechanics*, 656, 5–28, 2010. https://doi.org/10.1017/S0022112010001217

`EKFDMDc` uses EKF-DMD estimation with the DMD-with-control state equation. `SlidingWindowDMD` fits standard DMD independently to overlapping windows.

## License

MIT
