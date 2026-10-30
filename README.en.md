# SOFC Multiscale Modeling and Island Microgrid Research Code

English | [中文](README.md)

This project connects COMSOL cell modeling and field export, physics-informed neural network (PINN) surrogates, temperature sampling and lookup tables, capacity and economic analysis, and island microgrid energy management. The repository contains core research code, configurations, validation scripts, and the small datasets needed to use the released models. Full training data and the COMSOL reference model are distributed as Release assets.

## Recommended version and reproduction guides

The current documentation version is **v1.0.2**. The default deployed surrogate is **PINN v20 / final.pt**. See the [version selection guide](docs/VERSIONS.md) for the relationships between trainers, configurations, checkpoints, and lookup tables, and the [results index](docs/RESULTS_INDEX.md) for inputs, commands, and outputs associated with each result. These detailed guides are currently in Chinese. Full data assets remain available from the verified [v1.0.0 Release](https://github.com/IrisVow/sofc-multiscale-microgrid/releases/tag/v1.0.0).

## Repository layout

| Path | Contents |
|---|---|
| `outputs/pinn_sofc/` | v20 training, evaluation, ONNX export, network architecture, and physical residuals; `experiments/` contains separate physics-correction and interface-budget trainers |
| `outputs/microgrid_sys/` | System components, loads, dispatch, PINN coupling, capacity and economic analysis, sensitivity analysis, YAML configurations, and three temperature lookup tables |
| `outputs/pinndata/` | Operating-condition metadata, small samples, and full training-mesh coordinates; complete training tables are imported into the corresponding `merged/` directories |
| `work/` | COMSOL 2D design-of-experiments exports and shared project-path resolution |
| `outputs/verify3d/` | 3D model construction, solution, and diagnostic source code |
| `research/java/` | Native research tools for reactive transport, species, interface fluxes, meshes, and reference solutions |
| `research/test_*.py` | Physical equations, manufactured solutions, interface gradients, and species-permutation checks |
| `reference_results/` | Archived capacity results and the three-temperature-sampling dispatch comparison |
| `tools/` | Asset import, unified entry points, lookup-table reconstruction, and COMSOL compilation |
| `docs/` | Reproduction scope, path changes, provenance, and validation records |

The original `outputs/` module hierarchy is retained to preserve scientific implementations and import relationships. Runtime paths are resolved from the `.sofc-project` marker; configuration paths are relative to the repository or the corresponding module. Keep this marker in the repository root.

## Installation

Use Python 3.12. From the repository root:

```sh
python -m venv .venv
# Activate the virtual environment, then run:
python -m pip install -r requirements.txt
```

To match the validated Windows/CUDA environment, use **Python 3.12.13** and the pinned dependency file:

```sh
python -m pip install -r requirements-lock-windows-py312-cu121.txt
```

The lock file pins 37 direct and transitive dependencies and includes the official PyTorch CUDA 12.1 wheel index. See [environment details](docs/ENVIRONMENT.md) and `environment.snapshot.json` for the recorded system, Python, CUDA runtime, and package versions. `requirements.txt` retains dependency ranges for other environments. The pinned file records the environment used for validation; a new installation into an empty environment has not been tested.

CPU execution supports dispatch, capacity analysis, inference, and basic validation. A CUDA-enabled PyTorch environment is recommended for full PINN training. The historical semantic-interface validation requires CUDA. COMSOL steps require a separate COMSOL Multiphysics 6.3 installation with the relevant physics modules; COMSOL is not a Python dependency.

## Entry points that do not require the full training data

Run these commands from the repository root using the active Python interpreter:

```sh
python tools/run.py check
python tools/run.py dispatch 7 --pinn
python tools/run.py sampling
python tools/run.py capacity --pinn
python tools/run.py sensitivity
python tools/run.py train-smoke
```

`check` runs component, EMS, load, sizing, scaling, lookup-table, ONNX/PyTorch, inference API, capacity/economic, and four-species permutation checks. Performance assertions depend on the machine. The historical v18 model is not included, and its compatibility check is explicitly skipped.

`sampling` replays 13 seven-day runs: one baseline and four control modes for each of the three temperature definitions. It compares generation, main-fuel energy, GHG emissions, and final battery state of charge against `reference_results/ems_three_sampling/summary.csv`. Outputs are written to `outputs/microgrid_sys/results/sampling_replay/`. Use `--output` to select a new directory for subsequent runs.

`train-smoke` runs one epoch using deterministic samples from the actual data. It disables PDE terms and checks the training pipeline; it does not reproduce v20 model accuracy. Existing output directories are protected. Use `--run-dir runs/new_smoke` to choose a new location.

## Full data and model assets

| Release asset | Purpose |
|---|---|
| `sofc-training-data-v20.zip` | Complete merged steady and transient Parquet tables used by v20 |
| `sofc-comsol-reference.zip` | Original 2D reference model: `work/sofc1_FINAL2_solved.mph` |
| `sofc-advanced-validation.zip` | Historical epoch-30 checkpoint required for semantic-interface validation |

Download the required assets from the [v1.0.0 Release](https://github.com/IrisVow/sofc-multiscale-microgrid/releases/tag/v1.0.0) into `downloads/`, then run:

```sh
python tools/data_assets.py verify --asset training
python tools/data_assets.py import --asset training
python tools/data_assets.py import --asset comsol
python tools/data_assets.py import --asset advanced
```

Use `--directory` to select another local asset directory. The importer checks the size and SHA-256 digest of each archive and its contents. It stops if an existing destination file differs. This repository is private, so downloading assets requires an account with repository access. If a Release download URL is accessible to your client, `download --base-url` supports automatic download, verification, and import; otherwise download through the authenticated browser and import locally.

Full Parquet tables, `.mph` models, and download caches are excluded by `.gitignore`. Large data assets are kept in Releases rather than duplicated in the source tree.

## Full training and evaluation

After importing the training data, run the following from `outputs/pinn_sofc/`:

```sh
python train.py --config config_v20.yaml --init runs/stage3_pde_v19/best.pt --run-dir runs/new_v20
python evaluate.py --run_dir runs/stage3_pde_v20 --config config_v20.yaml --ckpt final.pt --pools steady_val steady_test transient_test
python export_onnx.py --config config_v20.yaml --ckpt runs/stage3_pde_v20/final.pt --out runs/stage3_pde_v20/new_export.onnx
```

The deployed v20 model uses `final.pt`. Its ONNX model and all three temperature lookup tables use the frozen deployment model. The v19 `best.pt` checkpoint is included to reproduce the historical v20 warm start. Released checkpoint configuration paths were relocated without changing model tensors. Training from random initialization and reproducing a historical warm-start protocol are different experiments.

Separate research trainers are located in `experiments/*/trainer/`. From the corresponding trainer directory, use `train.py --config ../configs/<configuration>.yaml --run-dir runs/new_experiment`. See the [version guide](docs/VERSIONS.md) and [reproduction scope](docs/REPRODUCIBILITY.md) for their roles. Historical checkpoints do not establish exact full-state resumption after code migration; use `--init` when starting a new experiment from existing model weights.

## Temperature lookup-table reconstruction

The repository contains three frozen lookup tables and the geometry of all 15,567 training-mesh points. Candidate temperature tables can be rebuilt independently of the complete training tables:

```sh
python tools/build_sampling_luts.py --sampling axial_101 --output outputs/microgrid_sys/results/lut_rebuild/axial101.npz
python tools/build_sampling_luts.py --sampling full_training_mesh --output outputs/microgrid_sys/results/lut_rebuild/full_mesh.npz
```

The reconstruction replaces only `tmax_k`, retaining the original operating axes, heat-source fields, and potential-drop field. Floating-point results can vary between devices. Existing deployment lookup tables are not overwritten.

## COMSOL

Run from the repository root. Set `COMSOL_HOME` to the local `Multiphysics` installation directory, or pass `--comsol-home` explicitly. For example, after setting `COMSOL_HOME`:

```sh
python tools/comsol.py --compile-only DoeAll
python tools/comsol.py DoeAll
python tools/comsol.py Build3D
```

Import the COMSOL reference asset before running the 2D DOE export. Model construction and solution produce new model/data files and may take substantial time. Diagnostic and refinement classes in `research/java/` generally require existing models, logs, and export paths; consult each class's `main` method for its arguments.

## Validation and study boundaries

The [validation record](docs/VALIDATION.md) documents 193 basic assertions, 32 physics/electrochemistry/interface checks, 1,000 randomized species-permutation cases, the 13-run dispatch replay, and successful compilation of 103 Java source files. A fresh extraction into a path containing spaces also passed the sample-training and ONNX/PyTorch entry points. These checks do not replace full PINN retraining or a new COMSOL solution, neither of which was performed during repository preparation.

Loads are synthetic representative island loads. Temperature limits and economic/emission parameters include engineering scenario assumptions. All three temperature tables are quasi-steady and omit thermal history. Temperatures outside the surrogate domain remain unknown. The historical controller can increase a low-power request when its strict-temperature search falls back to a fixed lower bound; this behavior is retained to reproduce the archived comparison.

Capacity and dispatch studies use different representative weeks and equipment configurations. Deployment, random-initialization pairs, and warm-start branches also have distinct protocols and purposes. Their results do not establish a jointly optimized capacity and dynamic thermal-constrained dispatch solution. See the [reproduction scope](docs/REPRODUCIBILITY.md) for the complete boundaries.
