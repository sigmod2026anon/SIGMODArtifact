# Mathematical Foundations of Poisoning Attacks on Linear Regression over Cumulative Distribution Functions

This repository contains the code for the paper "Mathematical Foundations of Poisoning Attacks on Linear Regression over Cumulative Distribution Functions". The code is based on [SOSD](https://github.com/learnedsystems/SOSD) (Search on Sorted Data Benchmark).

## Repository Structure (What is modified from SOSD)

### Modified Files
- `.gitignore`

### Added Files
- `Dockerfile`
- `docker_build.sh`
- `docker_run.sh`
- `docker_run_all.sh`
- `requirements.txt`
- `poisoning_projects/` - Contains all the poisoning attack implementation

## Quick Start

### Dataset Download (if using SOSD datasets)

```bash
./scripts/download.sh
```

### Simple Test

To run a simple test of the poisoning attack algorithms:

```bash
cd poisoning_projects/poisoning/scripts
./test.sh
```

This executes the content of `poisoning_projects/poisoning/cpp/apps/poisoning_quick_test_main.cpp`.

#### Expected Output Example

```
================================================================================
xs    : [0, 3, 7, 8]
lambda: 1
--------------------------------------------------------------------------------
loss (legitimate)            : 0.054878048780
loss (poisoned, greedy)      : 0.130841121495 (Time = 0.001ms)
loss (poisoned, brute force) : 0.130841121495 (Time = 0.002ms)
upper bound (binary search)  : 0.263888888889 (w* = 0.416667, Time = 0.000ms)
upper bound (golden section) : 0.263888888889 (w* = 0.416667, Time = 0.000ms)
upper bound (strict)         : 0.263888888889 (w* = 0.416667, Time = 0.000ms)
--------------------------------------------------------------------------------
poison values (greedy)      : [6]
poison values (brute force) : [6]
================================================================================

================================================================================
xs    : [0, 3, 7]
lambda: 1
--------------------------------------------------------------------------------
loss (legitimate)            : 0.004504504505
loss (poisoned, greedy)      : 0.100000000000 (Time = 0.000ms)
loss (poisoned, brute force) : 0.100000000000 (Time = 0.000ms)
upper bound (binary search)  : 0.159090909091 (w* = 0.363636, Time = 0.000ms)
upper bound (golden section) : 0.159090909091 (w* = 0.363636, Time = 0.000ms)
upper bound (strict)         : 0.159090909091 (w* = 0.363636, Time = 0.000ms)
--------------------------------------------------------------------------------
poison values (greedy)      : [1]
poison values (brute force) : [1]
================================================================================
```

## Full Experiments

### Using Docker

1. Build the Docker image:
```bash
./docker_build.sh
```

2. Run the container:
```bash
./docker_run.sh
```

3. Inside the Docker container, run the comprehensive experiments:
```bash
cd poisoning_projects/poisoning/scripts
./run_comprehensive_experiment.sh --all
```

Alternatively, you can run everything in one command:
```bash
./docker_run_all.sh --all
```

### Without Docker

```bash
cd poisoning_projects/poisoning/scripts
./run_comprehensive_experiment.sh --all
```

### Experiment Options

- Use `--all` for comprehensive experiments
- Use `--quick` for smaller-scale experiments

## Original SOSD Citation

This work builds upon the SOSD benchmark. If you use this code, please also cite the original SOSD papers:

```bibtex
@article{sosd-vldb,
  author    = {Ryan Marcus and
               Andreas Kipf and
               Alexander van Renen and
               Mihail Stoian and
               Sanchit Misra and
               Alfons Kemper and
               Thomas Neumann and
               Tim Kraska},
  title     = {Benchmarking Learned Indexes},
  journal   = {Proc. {VLDB} Endow.},
  volume    = {14},
  number    = {1},
  pages     = {1--13},
  year      = {2020}
}

@article{sosd-neurips,
  title={SOSD: A Benchmark for Learned Indexes},
  author={Kipf, Andreas and Marcus, Ryan and van Renen, Alexander and Stoian, Mihail and Kemper, Alfons and Kraska, Tim and Neumann, Thomas},
  journal={NeurIPS Workshop on Machine Learning for Systems},
  year={2019}
}
```
