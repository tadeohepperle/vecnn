# Master Thesis

Title: **Vantage-Point Tree to HNSW: Fast Index Construction for Approximate Nearest Neighbor Search in High-Dimensional Data**
University: TU Dortmund University
Author: Tadeo Hepperle

This file gives a quick an overview of important files and folders:

## `/vecnn`, the Rust crate

- rust crate with all the core logic
- `bin/compare.rs` can run benchmarks on SISAP LAION-2B data
- `slice_hnsw.rs` contains the current single threaded HNSW implementation
- `slice_hnsw_par.rs` has a multithreaded HNSW implementation
- `nn_descent.rs` has Relative NN-Descent implementation
- Stitching and VP-Tree ensemble methods are in `transition.rs`
- `schubert_distance.rs` contains distance functions provided by Prof. Dr. Erich Schubert from TU Dortmund.

## `/vecnnpy`, the Python wrapper

- wraps the `vecnn` rust crate and exposes a python interface.
- to use it, run: `python -m pip install -e .` (or replace the "." by "./my/path/to/vecnnpy")
- also exposes python interface to two other Rust HNSW libraries

## `/eval` some Helper scripts

- `get_data.py` can be run to download the SISAP datasets and convert them to binary (f32 slices)
- `eval.py` is a script used for benchmarking models. Make sure you run `python -m pip install -e ./path/to/vecnnpy` first, to install the `vecnn` python module.
- `results.py` uses mainly output from `compare.rs` to generate some tables and graphs (100K to 10M datasets).
- `res10m.py` uses output from `eval.py` to generate graphs and tables for benchmarks on the 10M dataset.
