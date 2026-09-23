# Ceres Playground

A C++ playground for learning and experimenting with [Ceres Solver](http://ceres-solver.org/), and nonlinear optimization using a reproducible Pixi environment.

The goal of this repository is to provide a simple environment where small optimization problems can be implemented, compiled, modified, and tested without installing Ceres and its dependencies system-wide.

## Requirements

- [Pixi](https://pixi.prefix.dev/latest/)

Ceres, Sophus, Eigen, CMake, Ninja, the C++ toolchain, and their required dependencies are managed through the Pixi environment.

## Get started

```bash
git clone https://github.com/terremat/ceres-playground.git
cd ceres-playground
pixi install
```

Pixi will create an isolated project environment and install the required development dependencies.

## Building

Configure and build the project through the provided Pixi task:

```bash
pixi run build
```

## Examples

The numbered files in `src/` introduce Ceres concepts in sequence:

- `01_hello_ceres.cpp` (`hello_ceres`): optimize one scalar with one residual using `AutoDiffCostFunction<..., 1, 1>`.
- `02_pose_estimation.cpp` (`pose_estimation`): optimize six pose parameters per camera with many 2D reprojection residuals using `AutoDiffCostFunction<..., 2, 6>`. Intrinsics, landmarks, and observations stay fixed.

Both lessons keep the cost functor, problem construction, solver options, and solve call visible in the executable. Shared types, geometry, synthetic data generation, RMSE metrics, and Rerun helpers live in `include/ceres_playground/`.

`examples/hello_rerun.cpp` (`hello_rerun`) is a separate visualization smoke test.

After building, run an example through the Pixi environment:

```bash
pixi run ./build/hello_ceres
pixi run ./build/pose_estimation
pixi run ./build/hello_rerun
```

The pose estimation and Rerun examples launch the Rerun viewer.
