// use std::{
//     cmp::Reverse, collections::BinaryHeap, collections::HashSet, sync::Arc, time::Instant, usize,
// };

// use rand::{seq::SliceRandom, thread_rng, Rng, SeedableRng};
// use rand_chacha::ChaCha20Rng;

// use crate::{
//     dataset::DatasetT,
//     distance::{Distance, DistanceTracker},
//     hnsw::{DistAnd, HnswParams},
//     slice_hnsw::{Layer, LayerEntry, SliceHnsw, MAX_LAYERS},
//     utils::{
//         extend_lifetime, extend_lifetime_mut, slice_binary_heap_arena, SliceBinaryHeap,
//         SlicesMemory, Stats,
//     },
// };

// #[derive(Debug)]
// pub struct NNDescentElement {
//     id: usize,
//     neighbors: SliceBinaryHeap<'static, Neighbor>,
// }

// #[derive(Debug, Clone)]
// pub struct Neighbor {
//     pub is_new: bool,
//     pub dist: f32,
//     pub idx: usize,
// }

// impl PartialEq for Neighbor {
//     fn eq(&self, other: &Self) -> bool {
//         self.idx == other.idx
//     }
// }
// impl Eq for Neighbor {}
// impl PartialOrd for Neighbor {
//     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//         self.dist.partial_cmp(&other.dist)
//     }
// }

// impl Ord for Neighbor {
//     fn cmp(&self, other: &Self) -> std::cmp::Ordering {
//         self.dist.total_cmp(&other.dist)
//     }
// }

// #[derive(Debug, Clone, Copy, PartialEq)]
// pub struct NNDescentParams {
//     pub m_max: usize,
//     pub iterations: usize,
//     pub distance: Distance,
//     pub p: f32, // fraction of K items sampled.
// }

// #[derive(Debug)]
// pub struct Entry {
//     id: usize,
//     neighbors: SliceBinaryHeap<'static, Neighbor>,
// }

// pub fn nn_descent_entries_arena(
//     ids: impl ExactSizeIterator<Item = usize>,
//     max_neighbors: usize,
// ) -> (SlicesMemory<Neighbor>, Vec<Entry>) {
//     let n_entries = ids.len();
//     let mut memory: SlicesMemory<Neighbor> = SlicesMemory::new(n_entries, max_neighbors);
//     let mut entries: Vec<Entry> = Vec::with_capacity(n_entries);
//     unsafe {
//         entries.set_len(n_entries);
//         for i in 0..n_entries {
//             let entry = entries.get_unchecked_mut(i);
//             entry.id = 0;
//             entry.neighbors.len = 0;
//             entry.neighbors.slice = memory.static_slice_at(i);
//         }
//     }
//     (memory, entries)
// }

// #[derive(Debug)]
// pub struct NNDescentGraph {
//     pub data: Arc<dyn DatasetT>,
//     pub neighbors_memory: SlicesMemory<Neighbor>,
//     pub entries: Vec<Entry>,
//     pub params: NNDescentParams,
//     pub build_stats: Stats,
// }

// impl NNDescentGraph {
//     pub fn new(data: Arc<dyn DatasetT>, params: NNDescentParams, seed: u64) -> Self {
//         let distance = DistanceTracker::new(params.distance);
//         let start_time = Instant::now();

//         let (neighbors_memory, mut entries) = nn_descent_entries_arena(0..data.len(), params.m_max);
//         nn_descent_init_random_neighbors(&*data, &mut entries, &distance, seed);
//         let mut buffers = NNDescentBuffers::new(seed);
//         for _ in 0..params.iterations {
//             nn_descent_iteration(&*data, &mut entries, &distance, params, &mut buffers);
//         }

//         let build_stats = Stats {
//             num_distance_calculations: distance.num_calculations(),
//             duration: start_time.elapsed(),
//         };

//         NNDescentGraph {
//             data,
//             neighbors_memory,
//             entries,
//             params,
//             build_stats,
//         }
//     }

//     pub fn into_single_layer_hnsw(self) -> SliceHnsw {
//         let layers: heapless::Vec<crate::slice_hnsw::Layer, MAX_LAYERS> = Default::default();

//         // copy over all the entries into an HNSW layer.
//         // Omits the `is_new` field for all neighbors, otherwise the neighbors and entry order stays the same.
//         let mut layer = crate::slice_hnsw::Layer::new(self.params.m_max);
//         layer.entries_cap = self.entries.len();
//         layer.allocate_neighbors_memory();
//         for e in self.entries.into_iter() {
//             let idx = layer.add_entry_assuming_allocated_memory(e.id, usize::MAX);
//             let hnsw_neighbors = &mut layer.entries[idx].neighbors;
//             unsafe {
//                 e.neighbors
//                     .map_into(hnsw_neighbors, |e| DistAnd(e.dist, e.idx));
//             }
//         }

//         let params = HnswParams {
//             level_norm_param: 0.0,
//             ef_construction: 0,
//             m_max: self.params.m_max,
//             m_max_0: self.params.m_max,
//             distance: self.params.distance,
//         };
//         SliceHnsw {
//             data: self.data,
//             layers,
//             params,
//             build_stats: self.build_stats,
//         }
//     }

//     /// n_eps = number of entry points (randomly chosen)
//     pub fn knn_search(&self, q_data: &[f32], k: usize, ef: usize, n_eps: usize) {
//         let mut visited: HashSet<usize> = HashSet::with_capacity(ef);
//         let mut frontier: BinaryHeap<Reverse<DistAnd<usize>>> = BinaryHeap::with_capacity(ef);
//         let mut found: BinaryHeap<DistAnd<usize>> = BinaryHeap::with_capacity(ef);
//         let distance = DistanceTracker::new(self.params.distance);

//         let mut rng =
//             ChaCha20Rng::seed_from_u64(unsafe { std::mem::transmute((q_data[0], n_eps as f32)) });
//         // start with num_eps
//         for _ in 0..n_eps {
//             let ep_idx = rng.gen_range(0..self.entries.len());
//             let ep_id = self.entries[ep_idx].id;
//             let ep_data = self.data.get(ep_id);
//             let ep_dist = distance.distance(ep_data, q_data);
//             visited.insert(ep_idx);
//             found.push(DistAnd(ep_dist, ep_idx));
//             frontier.push(Reverse(DistAnd(ep_dist, ep_idx)));
//         }

//         while frontier.len() > 0 {
//             let DistAnd(c_dist, c_idx) = frontier.pop().unwrap().0;
//             let worst_dist_found = found.peek().unwrap().0;
//             if c_dist > worst_dist_found {
//                 break;
//             };
//             for nei in self.entries[c_idx].neighbors.iter() {
//                 let nei_idx = nei.idx;
//                 if visited.insert(nei_idx) {
//                     // only jumps here if was not visited before (newly inserted -> true)
//                     let nei_id = self.entries[nei_idx].id;
//                     let nei_data = self.data.get(nei_id);
//                     let nei_dist_to_q = distance.distance(nei_data, q_data);

//                     if found.len() < ef {
//                         // always insert if found still has space:
//                         frontier.push(Reverse(DistAnd(nei_dist_to_q, nei_idx)));
//                         found.push(DistAnd(nei_dist_to_q, nei_idx));
//                     } else {
//                         // otherwise only insert, if it is better than the worst found element:
//                         let mut worst_found = found.peek_mut().unwrap();
//                         if nei_dist_to_q < worst_found.dist() {
//                             frontier.push(Reverse(DistAnd(nei_dist_to_q, nei_idx)));
//                             *worst_found = DistAnd(nei_dist_to_q, nei_idx)
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// pub fn nn_descent_init_random_neighbors(
//     data: &dyn DatasetT,
//     entries: &mut [Entry],
//     distance: &DistanceTracker,
//     seed: u64,
// ) {
//     let mut rng = ChaCha20Rng::seed_from_u64(seed);
//     let n_entries = entries.len();
//     for idx in 0..entries.len() {
//         let entry = extend_lifetime_mut(&mut entries[idx]);
//         let entry_data = data.get(entry.id);
//         while entry.neighbors.len() < entry.neighbors.capacity() {
//             let mut nei_idx = rng.gen_range(0..n_entries - 1); // -1 bc. the range should be one shorter to skip idx, see below.
//             if nei_idx >= idx {
//                 nei_idx += 1; // ensures that random_idx != except_idx
//             }
//             // reroll random idx if duplicate: (maybe irrelevant though.)
//             if entry.neighbors.iter().any(|e| e.idx == nei_idx) {
//                 continue;
//             }
//             // lower idx will have come first and always have full neighbors already
//             let nei_entry = extend_lifetime_mut(&mut entries[nei_idx]);
//             let nei_entry_data = data.get(nei_entry.id);
//             let dist = distance.distance(entry_data, nei_entry_data);
//             entry.neighbors.insert_asserted(Neighbor {
//                 is_new: true,
//                 dist,
//                 idx: nei_idx,
//             });
//             if nei_idx > idx {
//                 nei_entry.neighbors.insert_asserted(Neighbor {
//                     is_new: true,
//                     dist,
//                     idx,
//                 });
//             }
//         }
//     }
// }

// // buffers used per vertex.
// pub struct NNDescentBuffers {
//     old: Vec<usize>,
//     new: Vec<usize>,
//     old_reverse: Vec<usize>,
//     new_reverse: Vec<usize>,
//     rng: ChaCha20Rng,
// }

// impl NNDescentBuffers {
//     pub fn new(seed: u64) -> Self {
//         let rng = ChaCha20Rng::seed_from_u64(seed);
//         NNDescentBuffers {
//             old: vec![],
//             new: vec![],
//             old_reverse: vec![],
//             new_reverse: vec![],
//             rng,
//         }
//     }
//     pub fn clear(&mut self) {
//         self.old.clear();
//         self.new.clear();
//         self.old_reverse.clear();
//         self.new_reverse.clear();
//     }
// }

// pub fn nn_descent_iteration(
//     data: &dyn DatasetT,
//     entries: &mut [Entry],
//     distance: &DistanceTracker,
//     params: NNDescentParams,
//     buffers: &mut NNDescentBuffers,
// ) {
//     let mut pK = (params.p * params.m_max as f32).round() as usize;
//     let mut update_counter: usize = 0;

//     for idx in 0..entries.len() {
//         buffers.clear();
//         let entry = &mut entries[idx];
//         let neighbors = &mut entry.neighbors;

//         // sort neighbors into two lists: old and new:
//         for nei in neighbors.iter() {
//             if nei.is_new {
//                 buffers.new.push(nei.idx);
//             } else {
//                 buffers.old.push(nei.idx);
//             }
//         }
//         // ut new_neighbors to max pK neighbors (random sampling):
//         buffers.new.shuffle(&mut buffers.rng);
//         buffers.new.truncate(pK);
//         // mark all sampled new_neighbors in neighbors as false: (could be faster if buffers.new would remember index into this neighbors list or something...)
//         for e in unsafe { neighbors.as_mut_slice().iter_mut() } {
//             if buffers.new.contains(&e.idx) {
//                 e.is_new = false;
//             }
//         }
//         // collect all the neig
//         for &nei in buffers.old.iter() {
//             let nei_neighbors = &entries[nei].neighbors;
//             for nei_nei in nei_neighbors.iter() {
//                 buffers.old_reverse.push(nei_nei.idx);
//             }
//         }
//         for &nei in buffers.new.iter() {
//             let nei_neighbors = &entries[nei].neighbors;
//             for nei_nei in nei_neighbors.iter() {
//                 buffers.new_reverse.push(nei_nei.idx);
//             }
//         }
//         buffers.old_reverse.sort();
//         buffers.new_reverse.sort();
//         remove_duplicates_for_sorted(&mut buffers.old_reverse);
//         remove_duplicates_for_sorted(&mut buffers.new_reverse);

//         // let old_reverse = reverse(old_neighbors)
//         // let new_reverse = reverse(new_neighbors)

//         // old_neighbors: add sample(old_reverse, pK)
//         // new_neighbors: add sample(new_reverse, pK)

//         // for u1

//         /*
//         idx_old_neighbors = idx_old_neighbors and reverse(idx_old_neighbors)

//         for each pair u1, u2 in idx_new_neighbors

//          */
//     }
// }

// /// Credit: Erik Thordsen (https://www-ai.cs.tu-dortmund.de/PERSONAL/thordsen.html)
// pub fn remove_duplicates_for_sorted(neighbors: &mut Vec<usize>) {
//     if neighbors.len() == 0 {
//         return;
//     }
//     // Last index of items to keep
//     let mut target: usize = 0;
//     // Identifier of the last item to keep for comparisons
//     let mut target_idx: usize = neighbors[0];
//     for i in 1..neighbors.len() {
//         unsafe {
//             let i_element = *neighbors.get_unchecked(i);
//             if i_element != target_idx {
//                 target += 1;
//                 target_idx = i_element;
//                 // Move element at i to target (overwriting duplicated between i and target):
//                 *neighbors.get_unchecked_mut(target) = i_element;
//             }
//         }
//     }
//     neighbors.truncate(target + 1);
// }
