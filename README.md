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