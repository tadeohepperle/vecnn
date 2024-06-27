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

#[macro_export]
macro_rules! track {
    ($event:expr) => {{
        #[cfg(feature = "tracking")]
        {
            use $crate::tracking::Event::*;
            $crate::tracking::push_event($event);
        }
    }};
}

#[cfg(feature = "tracking")]
pub mod tracking {
    use std::cell::UnsafeCell;

    thread_local! {
        pub static EVENTS: UnsafeCell<Vec<Event>> = const { UnsafeCell::new(vec![]) };
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

    pub fn push_event(event: Event) {
        EVENTS.with(|e| {
            let events = unsafe { &mut *e.get() };
            events.push(event);
        })
    }

    pub fn clear_events() -> Vec<Event> {
        EVENTS.with(|e| {
            let events = unsafe { &mut *e.get() };
            std::mem::take(events)
        })
    }
}
