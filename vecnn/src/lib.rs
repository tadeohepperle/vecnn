#![feature(const_trait_impl)]
#![feature(binary_heap_as_slice)]
#![feature(iter_array_chunks)]

pub mod dataset;
pub mod distance;
pub mod nn_descent;
pub mod transition;
pub mod utils;
pub mod vp_tree;

pub type Float = f32;
pub mod hnsw;
