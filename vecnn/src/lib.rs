#![feature(const_trait_impl)]
#![feature(binary_heap_as_slice)]
#![feature(iter_array_chunks)]
#![feature(const_collections_with_hasher)]

pub mod const_hnsw;
pub mod dataset;
pub mod distance;
mod nn_descent;
pub mod relative_nn_descent;
pub mod schubert_distance;
pub mod slice_hnsw;
pub mod slice_hnsw_par;
pub mod transition;
pub mod utils;
pub mod vp_tree;

pub type Float = f32;
pub mod hnsw;

#[macro_export]
macro_rules! if_tracking {
    ($($tt:tt)+) => {{
        #[cfg(feature = "tracking")]
        {
            use $crate::tracking::*;
            $($tt)+
        }
    }};
}
#[cfg(feature = "tracking")]
pub mod tracking {

    use std::cell::UnsafeCell;

    use std::collections::HashMap;
    use std::ops::{Deref, DerefMut};

    pub struct Tracking;

    impl Deref for Tracking {
        type Target = TrackingData;

        fn deref(&self) -> &Self::Target {
            TRACKING.with(|e| unsafe { &*e.get() })
        }
    }

    impl DerefMut for Tracking {
        fn deref_mut(&mut self) -> &mut Self::Target {
            TRACKING.with(|e| unsafe { &mut *e.get() })
        }
    }

    #[derive(Debug, Clone, Default)]
    pub struct TrackingData {
        pub events: Vec<Event>,
        pub pt_metadata: HashMap<usize, PtMeta>,
        pub edge_metadata: HashMap<(usize, usize), EdgeMeta>,
    }

    #[derive(Debug, Clone, Default)]
    pub struct PtMeta {
        pub chunk: usize,
        pub is_pos_center: bool,
        pub is_neg_random: bool,
        pub is_neg_cand: bool,
        pub annotation: Option<String>,
        pub chunk_on_level: Vec<usize>,
    }

    #[derive(Debug, Clone, Default)]
    pub struct EdgeMeta {
        pub is_neg_to_pos: bool,
        pub is_pos_to_neg: bool,
    }

    impl TrackingData {
        pub fn add_event(&mut self, event: Event) {
            self.events.push(event);
        }
        pub fn clear(&mut self) -> Self {
            std::mem::take(self)
        }
        pub fn pt_meta(&mut self, pt: usize) -> &mut PtMeta {
            self.pt_metadata.entry(pt).or_default()
        }
        pub fn edge_meta(&mut self, a: usize, b: usize) -> &mut EdgeMeta {
            self.edge_metadata.entry((a, b)).or_default()
        }
    }

    thread_local! {
        pub static TRACKING: UnsafeCell<TrackingData> = UnsafeCell::new(Default::default());
    }

    #[derive(Debug, Clone, Copy)]
    pub enum Event {
        Point {
            id: usize,
            level: usize,
        },
        EdgeHorizontal {
            from: usize,
            to: usize,
            level: usize,
            comment: &'static str,
        },
        EdgeDown {
            from: usize,
            upper_level: usize,
        },
    }
}
