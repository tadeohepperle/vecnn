# Master Thesis

Title: **Vantage-Point Tree to HNSW: Fast Index Construction for Approximate Nearest Neighbor Search in High-Dimensional Data**
University: TU Dortmund University
Author: Tadeo Hepperle

This file gives a quick an overview of important (code and data) files and folders:

## `/vecnn`, the Rust crate

- rust crate with all the core logic
- `bin/compare.rs` can run benchmarks on SISAP LAION-2B data for all kinds of models from the `vecnn` crate (HNSW, RNN-Descent, VP-tree ensemble, stitching, ...). Also computes real 1000 nearest neighbors and caches them on disk to avoid these costly operations.
- `slice_hnsw.rs` contains the current single threaded HNSW implementation
- `slice_hnsw_par.rs` has a multithreaded HNSW implementation
- `nn_descent.rs` has Relative NN-Descent implementation
- Stitching and VP-Tree ensemble methods are in `transition.rs`
- `schubert_distance.rs` contains distance functions provided by Prof. Dr. Erich Schubert from TU Dortmund.

## `/vecnnpy`, the Python wrapper

- wraps the `vecnn` rust crate and exposes a python interface.
- to use it, run: `python -m pip install -e .` (or replace the "." by "./my/path/to/vecnnpy")
- also exposes python interface to two other Rust HNSW libraries

## `/eval`, some Helper scripts

- `get_data.py` can be run to download the SISAP datasets and convert them to binary (f32 slices)
- `eval.py` is a script used for benchmarking models. Make sure you run `python -m pip install -e ./path/to/vecnnpy` first, to install the `vecnn` python module.
- `results.py` uses mainly output from `compare.rs` to generate some tables and graphs (100K to 10M datasets).
- `res10m.py` uses output from `eval.py` to generate graphs and tables for benchmarks on the 10M dataset.

- Note: better create a virtual environment in `/eval` with `python -m venv venv` and then install our `vecnn` Rust module by `python -m pip install -e ../vecnnpy`.

## `a10m_filtered_all_final.csv`, collected data

- benchmarks on the 10M datasets are collected into this one big file. Represents weeks of experiments run on a Uni Server from TU Dortmund.
- other data, e.g. experiments for parameter tuning on the 100K subsets can be found in the `eval/thesis_experiments` directory.
