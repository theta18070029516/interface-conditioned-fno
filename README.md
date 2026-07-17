# Feature-conditioned spectral operators for 1D advection

This project compares linear Fourier operators for interface-aware advection on

$$
u_t + c u_x = 0,
\qquad x \in [-1,1],
\quad t \in [0,1],
\quad c=0.5.
$$

The primary models are

$$
\mathcal T_{\mathrm{shared}}u_0
=a\mathcal Ku_0+b\gamma_1\mathcal K(\gamma_0u_0),
$$

and

$$
\mathcal T_{\mathrm{dual}}u_0
=\mathcal K_0u_0+\gamma_1\mathcal K_1(\gamma_0u_0).
$$

The parameter-matched generic two-channel baseline is

$$
\mathcal T_{\mathrm{2ch}}(u_0,\gamma_0)
=\mathcal K_u u_0+\mathcal K_\gamma\gamma_0.
$$

It is a standard single-layer linear two-input-channel, one-output-channel
Fourier operator. It deliberately does not use the structured product
$\gamma_0u_0$ or the output gate $\gamma_1$. Its two independent multipliers
give it exactly the same parameter count as `dual`, isolating the benefit of
the Dual factorization from a generic extra input channel.

The linear hidden-width control is

$$
h_1=\mathcal K_1u_0,
\qquad
h_2=\mathcal K_2u_0,
\qquad
\mathcal T_{\mathrm{width2}}u_0=h_1+h_2.
$$

Because

$$
\mathcal K_1u_0+\mathcal K_2u_0
=(\mathcal K_1+\mathcal K_2)u_0,
$$

`hidden_width_2` has twice the stored multiplier parameters but exactly the
same linear function class as `plain`. It is a negative control for whether
parameter duplication alone explains any gain. The two branches are
non-identifiable, so fitting uses the minimum-norm split
$\mathcal K_1=\mathcal K_2=\mathcal K_{\mathrm{eff}}/2$. Its per-branch ridge
coefficient is doubled, making the effective regularization on
$\mathcal K_{\mathrm{eff}}$ equal to the `plain` ridge coefficient.

The secondary ablation `shared_learnable_rho` keeps the single-layer shared
operator but learns its cross-interface coupling:

$$
\rho=\mathrm{sigmoid}(\eta),
\qquad
a=\frac{1+\rho}{2},
\qquad
b=\frac{1-\rho}{2}.
$$

It jointly optimizes the scalar $\eta$ and the shared Fourier multiplier. The
fixed-$\rho$ `shared` model remains unchanged and is still fit by deterministic
ridge regression. The learned model saves both the final `rho` and its
unconstrained `rho_eta` value in `models/*.npz`. Each fit learns one global
$\rho$ for one stage, seed, and retained-mode budget; it is not sample-dependent.

Here $\gamma_0$ marks the initial interface and
$\gamma_1(x)=\gamma_0(x-cT)$ marks the transported interface. The benchmark
uses an inflow-outflow formulation so every interior two-plateau sample has one
physical jump. Samples with $s=-1$ or $s=1$ use constant $\gamma$ and are
continuous controls.

## What is model input?

`plain`, `wide_plain`, and `hidden_width_2` use only `u0`. `two_channel_fno`
uses the two channels `u0` and `gamma0`. The structured `shared`,
`shared_learnable_rho`, and `dual` models use `u0`, `gamma0`, and `gamma1` in
their prescribed multiplications. The stored metadata (`s`, `J`, `c`, `T`,
smooth coefficients, endpoint flags, seeds) is used to regenerate samples and
stratify metrics; it is not passed to the operator.

## Local verification

The configured local environment is CPU-only JAX:

```powershell
C:\Users\Hollon\miniconda3\envs\jax\python.exe -m unittest discover -s tests -v
C:\Users\Hollon\miniconda3\envs\jax\python.exe -m advection_fno.experiment --preset smoke --stage all --output-dir results/smoke
```

The smoke preset is intentionally small. It exercises both data stages, all
models, the $\rho$ scan, interface diagnostics, and report generation.

## Formal GPU run

On a Linux NVIDIA server, install a GPU-enabled JAX build appropriate for the
driver. The current official JAX documentation recommends pip-provided CUDA
wheels, for example `jax[cuda13]` or `jax[cuda12]`. Check the live instructions
before installation:

<https://docs.jax.dev/en/latest/installation.html>

Verify that JAX sees a GPU:

```bash
python -c "import jax; print(jax.devices())"
```

Then run the registered experiment:

```bash
python -m advection_fno.experiment \
  --preset full \
  --stage all \
  --output-dir results/full \
  --save-data \
  --x64
```

`--x64` improves the conditioning audit for deterministic ridge fitting but can
be slower on GPUs with weak FP64 throughput. Run once without it only as a
performance diagnostic; use the same precision setting for every compared
model.

## Outputs

- `manifest.json`: configuration, environment, devices, seeds, and duration.
- `metrics.csv`: one row per evaluated sample.
- `summary.csv`: grouped mean and median metrics.
- `comparisons.csv`: paired bootstrap comparisons against plain and
  parameter-matched baselines, including `dual` versus `two_channel_fno` and
  `dual` versus `hidden_width_2`.
- `models/*.npz`: learned complex Fourier multipliers and fit diagnostics.
- `datasets/*.npz`: generated arrays when `--save-data` is supplied.
- `figures/*.png`: representative predictions and error-vs-mode curves.

The primary acceptance rule is a minimum 20% reduction in both interface MAE
and overshoot, with a paired 95% bootstrap interval above zero, while preserving
continuous endpoint performance.
