use std::{
    cell::UnsafeCell,
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
    sync::Arc,
    time::{Duration, Instant},
};

use arrayvec::ArrayVec;
use heapless::binary_heap::{Max, Min};
use rand::{thread_rng, Rng};
use rand_chacha::ChaChaRng;

macro_rules! track {
    ($event:expr) => {{
        #[cfg(feature = "tracking")]
        {
            use tracking::Event::*;
            tracking::push_event($event)
        }
    }};
}

use crate::{
    dataset::DatasetT,
    distance::{DistanceT, DistanceTracker, SquaredDiffSum},
    vp_tree::Stats,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HnswParams {
    /// normalization factor for level generation
    /// Influences the chance of at which level a point is interted.
    pub level_norm_param: f32,
    pub ef_construction: usize,
    pub m_max: usize,
    pub m_max_0: usize,
}

impl HnswParams {
    /// Returns max number of connections allowed on layer l
    #[inline(always)]
    fn m_max_on_level(&self, l: usize) -> usize {
        if l == 0 {
            self.m_max_0
        } else {
            self.m_max
        }
    }
}

#[derive(Debug, Clone)]
pub struct Hnsw {
    pub params: HnswParams,
    pub data: Arc<dyn DatasetT>,
    pub layers: Vec<Layer>,
    pub build_stats: Stats,
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub level: usize,
    pub entries: Vec<LayerEntry>,
}

type ID = u32;
const M: usize = 40;

#[derive(Debug, Clone)]
pub struct LayerEntry {
    pub id: ID,
    /// pos where we can find this entry at a lower level.
    /// insignificat on level 0, just set to u32::MAX.
    pub lower_level_idx: u32,
    /// a Max-Heap, such that we can easily pop off the item with the largest distance to make space.
    /// DistAnd<u32> is distances to, and idx's of neighbors
    pub neighbors: heapless::BinaryHeap<DistAnd<u32>, Max, M>, // the u32 stores the index in the layer
}
impl LayerEntry {
    fn new(id: u32, lower_level_idx: u32) -> LayerEntry {
        LayerEntry {
            id,
            lower_level_idx,
            neighbors: Default::default(),
        }
    }
}

impl Hnsw {
    pub fn new(data: Arc<dyn DatasetT>, params: HnswParams) -> Self {
        construct_hnsw(data, params)
    }
    pub fn new_empty(data: Arc<dyn DatasetT>, params: HnswParams) -> Self {
        Hnsw {
            params,
            data,
            layers: vec![],
            build_stats: Stats::default(),
        }
    }

    pub fn knn_search(&self, q_data: &[f32], k: usize) -> (Vec<SearchLayerRes>, Stats) {
        assert_eq!(q_data.len(), self.data.dims());

        let distance = DistanceTracker::new(SquaredDiffSum::distance);
        let start = Instant::now();

        let mut ep_idx_in_layer = 0;
        for i in (1..self.layers.len()).rev() {
            let layer = &self.layers[i];
            let res =
                closest_point_in_layer(layer, &*self.data, q_data, ep_idx_in_layer, &distance);
            ep_idx_in_layer = res.idx_in_lower_layer;
            track!(EdgeDown {
                from: res.id,
                upper_level: i,
            });
        }

        let ef = self.params.ef_construction.max(k);
        let mut out = SearchBuffers::new(ef);
        closests_points_in_layer(
            &self.layers[0],
            &*self.data,
            q_data,
            &[ep_idx_in_layer],
            ef, // todo! maybe not right?
            &mut out,
            &distance,
        );
        let mut results: Vec<SearchLayerRes> = vec![];
        select_neighbors(&self.layers[0], &mut out.found, k, &mut results);
        let stats = Stats {
            num_distance_calculations: distance.num_calculations(),
            duration: start.elapsed(),
        };
        (results, stats)
    }
}

fn construct_hnsw(data: Arc<dyn DatasetT>, params: HnswParams) -> Hnsw {
    let tracker = DistanceTracker::new(SquaredDiffSum::distance);
    let start = Instant::now();

    let mut hnsw = Hnsw::new_empty(data, params);
    let len = hnsw.data.len();
    if len == 0 {
        return hnsw;
    }

    // insert a first layer with a first entry
    let mut entries = Vec::with_capacity(len);
    entries.push(LayerEntry {
        id: 0,
        lower_level_idx: u32::MAX,
        neighbors: Default::default(),
    });
    hnsw.layers.push(Layer {
        level: hnsw.layers.len(),
        entries,
    });
    // insert the rest of the points one by one
    for id in 1..len as u32 {
        insert(&mut hnsw, id, &tracker);
    }

    hnsw.build_stats = Stats {
        num_distance_calculations: tracker.num_calculations(),
        duration: start.elapsed(),
    };

    hnsw
}

fn insert(hnsw: &mut Hnsw, q: ID, distance: &DistanceTracker) {
    let HnswParams {
        level_norm_param,
        ef_construction,
        m_max,
        m_max_0,
    } = hnsw.params;

    let q_data = hnsw.data.get(q as usize);
    // /////////////////////////////////////////////////////////////////////////////
    // Phase 0: insert the element on all levels (with empty neighbors)
    // /////////////////////////////////////////////////////////////////////////////

    let top_l = hnsw.layers.len() - 1; // (previous top l)
    let insert_l = pick_level(level_norm_param);
    let mut lower_level_idx: u32 = u32::MAX;
    for l in 0..=insert_l {
        let entry = LayerEntry::new(q, lower_level_idx);
        if let Some(layer) = hnsw.layers.get_mut(l) {
            lower_level_idx = layer.entries.len() as u32;
            layer.entries.push(entry);
        } else {
            let layer = Layer {
                level: hnsw.layers.len(),
                entries: vec![entry],
            };
            hnsw.layers.push(layer);
            lower_level_idx = 0;
        }
    }

    // /////////////////////////////////////////////////////////////////////////////
    // Phase 1: find the idx of the entry point at level `insert_l`
    // /////////////////////////////////////////////////////////////////////////////

    // this loop only runs, if insert_l < top_l, otherwise, the entry point is just the first point of the highest level.
    let ep_l = top_l.min(insert_l);
    let mut ep_idx_at_ep_l = 0;
    for l in (insert_l + 1..=top_l).rev() {
        let res = closest_point_in_layer(
            &hnsw.layers[l],
            &*hnsw.data,
            q_data,
            ep_idx_at_ep_l,
            distance,
        );
        ep_idx_at_ep_l = res.idx_in_lower_layer;
    }

    // /////////////////////////////////////////////////////////////////////////////
    // Phase 2
    // /////////////////////////////////////////////////////////////////////////////

    let mut ep_idxs_in_layer = Vec::with_capacity(ef_construction); // ep
    ep_idxs_in_layer.push(ep_idx_at_ep_l);
    let mut search_buffers: SearchBuffers = SearchBuffers::new(ef_construction);
    let mut neighbors_out: Vec<SearchLayerRes> = vec![]; // neighbors in paper
    for l in (0..=ep_l).rev() {
        let layer = &mut hnsw.layers[l];
        closests_points_in_layer(
            layer,
            &*hnsw.data,
            q_data,
            &ep_idxs_in_layer,
            hnsw.params.ef_construction,
            &mut search_buffers,
            distance,
        );
        select_neighbors(layer, &mut search_buffers.found, M, &mut neighbors_out);
        // add bidirectional connections from neighbors to q at layer l:
        let idx_of_q_in_l = layer.entries.len() as u32 - 1;
        let m_max = hnsw.params.m_max_on_level(l);
        for n in neighbors_out.iter() {
            // add connection from q to n:
            layer.entries[idx_of_q_in_l as usize]
                .neighbors
                .push(DistAnd {
                    i: n.idx_in_layer,
                    dist: n.d_to_q,
                })
                .expect("should have space.");

            // add connection from n to q:
            let n_neighbors = &mut layer.entries[n.idx_in_layer as usize].neighbors;
            if n_neighbors.len() < m_max {
                n_neighbors
                    .push(DistAnd {
                        i: idx_of_q_in_l,
                        dist: n.d_to_q,
                    })
                    .expect("should have space too");
            } else {
                // if all neighbors in n_neighbors are closer already, dont add connection from n to q:
                let max_d = n_neighbors.peek().unwrap().dist;
                if max_d > n.d_to_q {
                    // because this is a max heap, pop removes the item with the greatest distance.
                    n_neighbors.pop().unwrap();
                    n_neighbors
                        .push(DistAnd {
                            i: idx_of_q_in_l,
                            dist: n.d_to_q,
                        })
                        .unwrap();
                }
            }
        }

        // set new ep_idxs_in_layer:
        ep_idxs_in_layer.clear();
        for e in search_buffers.found.iter() {
            ep_idxs_in_layer.push(e.i)
        }
    }
}

pub fn pick_level(level_norm_param: f32) -> usize {
    let f = thread_rng().gen::<f32>();
    (-f.ln() * level_norm_param).floor() as usize
}

#[test]
fn testlevel() {
    for i in 0..100 {
        println!("{}", pick_level(10.0))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SearchLayerRes {
    pub idx_in_layer: u32,
    pub idx_in_lower_layer: u32,
    pub id: ID,
    pub d_to_q: f32,
}

struct SearchBuffers {
    visited_idxs: HashSet<u32>,
    /// we need to be able to extract the closest element from this (so we use Reverse<IdxAndDist> to have a min-heap)
    candidates: BinaryHeap<Reverse<DistAnd<u32>>>,
    /// we need to be able to extract the furthest element from this: this is a max heap, the root is the max distance.
    found: BinaryHeap<DistAnd<u32>>,
}

impl SearchBuffers {
    fn new(capacity: usize) -> Self {
        SearchBuffers {
            visited_idxs: HashSet::with_capacity(capacity),
            candidates: BinaryHeap::with_capacity(capacity),
            found: BinaryHeap::with_capacity(capacity),
        }
    }

    fn initialize(&mut self, ep_idxs_in_layer: &[u32], idx_to_dist: impl Fn(u32) -> f32) {
        self.visited_idxs.clear();
        self.candidates.clear();
        self.found.clear();
        for idx_in_layer in ep_idxs_in_layer.iter().copied() {
            let dist = idx_to_dist(idx_in_layer);
            self.visited_idxs.insert(idx_in_layer);
            self.candidates.push(Reverse(DistAnd {
                i: idx_in_layer,
                dist,
            }));
            self.found.push(DistAnd {
                i: idx_in_layer,
                dist,
            })
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DistAnd<T: PartialEq + Copy> {
    pub dist: f32,
    pub i: T,
}

impl<T: PartialEq + Copy> PartialEq for DistAnd<T> {
    fn eq(&self, other: &Self) -> bool {
        self.i == other.i && self.dist == other.dist
    }
}
impl<T: PartialEq + Copy> Eq for DistAnd<T> {}
impl<T: PartialEq + Copy> PartialOrd for DistAnd<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}
impl<T: PartialEq + Copy> Ord for DistAnd<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.total_cmp(&other.dist)
    }
}

/// greedy routing trough the graph, going to the closest neighbor all the time.
fn closest_point_in_layer(
    layer: &Layer,
    data: &dyn DatasetT,
    q_data: &[f32],
    ep_idx_in_layer: u32,
    distance: &DistanceTracker,
) -> SearchLayerRes {
    // let visited_idxs: HashSet<usize> = HashSet::new(); // prob. not needed???

    // initialize best entry to the entry point (at ep_idx_in_layer)
    let mut best_entry_idx_in_layer = ep_idx_in_layer;
    let mut best_entry = &layer.entries[best_entry_idx_in_layer as usize];
    let mut best_entry_d = distance.distance(data.get(best_entry.id as usize), q_data);

    // iterate over all neighbors of best_entry, go to the one with lowest distance to q.
    // if none of them better than current best entry return (greedy routing).
    loop {
        track!(Point {
            id: best_entry.id,
            level: layer.level
        });
        #[cfg(feature = "tracking")]
        let best_entry_id = best_entry.id;

        let mut found_a_better_neighbor = false;
        for idx_and_dist in best_entry.neighbors.iter() {
            let n = &layer.entries[idx_and_dist.i as usize];
            let n_d = distance.distance(data.get(n.id as usize), q_data);
            if n_d < best_entry_d {
                best_entry_d = n_d;
                best_entry = n;
                found_a_better_neighbor = true;
                best_entry_idx_in_layer = idx_and_dist.i;
            }
        }
        if !found_a_better_neighbor {
            return SearchLayerRes {
                idx_in_layer: best_entry_idx_in_layer,
                idx_in_lower_layer: best_entry.lower_level_idx,
                id: best_entry.id,
                d_to_q: best_entry_d,
            };
        }
        track!(EdgeHorizontal {
            from: best_entry_id,
            to: best_entry.id,
            level: layer.level
        });
    }
}

/// after doing this, out.found will contain the relevant found points.
fn closests_points_in_layer(
    layer: &Layer,
    data: &dyn DatasetT,
    q_data: &[f32],
    ep_idxs_in_layer: &[u32],
    ef: usize, // max number of found items
    out: &mut SearchBuffers,
    distance: &DistanceTracker,
) {
    // #[cfg(feature = "tracking")]
    // let mut track_to_idx_from_idx: HashMap<u32, u32> = HashMap::new();
    out.initialize(ep_idxs_in_layer, |idx| {
        let id = layer.entries[idx as usize].id;
        distance.distance(data.get(id as usize), q_data)
    });

    while out.candidates.len() > 0 {
        let c = out.candidates.pop().unwrap(); // remove closest element.
        let mut f = *out.found.peek().unwrap();
        if c.0.dist > f.dist {
            break; // all elements in found are evaluated (see paper).
        }
        let c_entry = &layer.entries[c.0.i as usize];
        for idx_and_dist in c_entry.neighbors.iter() {
            let idx_in_layer = idx_and_dist.i;
            if out.visited_idxs.insert(idx_in_layer) {
                let n_entry = &layer.entries[idx_in_layer as usize];
                let n_data = data.get(n_entry.id as usize);
                let dist = distance.distance(q_data, n_data);
                f = *out.found.peek().unwrap();
                if dist < f.dist || out.found.len() < ef {
                    out.candidates.push(Reverse(DistAnd {
                        i: idx_in_layer,
                        dist,
                    }));

                    if out.found.len() < ef {
                        out.found.push(DistAnd {
                            i: idx_in_layer,
                            dist,
                        });

                        // #[cfg(feature = "tracking")]
                        // track_to_idx_from_idx.insert(idx_in_layer, c.0.idx_in_layer);
                    } else {
                        // compare dist to the currently furthest away,
                        // if further than this dist, kick it out and insert the new one instead.
                        if dist < f.dist {
                            out.found.pop().unwrap();
                            out.found.push(DistAnd {
                                i: idx_in_layer,
                                dist,
                            });

                            // #[cfg(feature = "tracking")]
                            // track_to_idx_from_idx.insert(idx_in_layer, c.0.idx_in_layer);
                        }
                    }
                }
            }
        }
    }

    // #[cfg(feature = "tracking")]
    // {
    //     for e in out.found.iter() {
    //         let to = layer.entries[e.idx_in_layer as usize].id;
    //         let Some(from_idx) = track_to_idx_from_idx.get(&e.idx_in_layer) else {
    //             continue;
    //         };
    //         let from = layer.entries[*from_idx as usize].id;
    //         track!(EdgeHorizontal {
    //             from,
    //             to,
    //             level: layer.level
    //         })
    //     }
    // }
}

/// todo! what if less neighbors there? will fail??
fn select_neighbors(
    layer: &Layer,
    candidates: &mut BinaryHeap<DistAnd<u32>>, // a max-heap where the root is the largest-dist element.
    n: usize,
    out: &mut Vec<SearchLayerRes>,
) {
    while candidates.len() > n {
        // removes the furthest element from candidates, leaving only the n closest ones in it.
        candidates.pop();
    }

    out.clear();
    for c in candidates.iter() {
        let entry = &layer.entries[c.i as usize];
        out.push(SearchLayerRes {
            idx_in_layer: c.i,
            idx_in_lower_layer: entry.lower_level_idx,
            id: entry.id,
            d_to_q: c.dist,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::Hnsw;

    #[test]
    fn build_hnsw() {
        let data = Arc::new(vec![
            [0.0, 0.0],
            [2.0, 1.0],
            [2.0, 6.0],
            [3.0, 6.0],
            [4.0, 6.0],
            [7.0, 7.0],
        ]);

        let hnsw = Hnsw::new(
            data,
            super::HnswParams {
                level_norm_param: 2.0,
                ef_construction: 4,
                m_max: 2,
                m_max_0: 3,
            },
        );

        dbg!(hnsw);
    }
}

#[cfg(feature = "tracking")]
pub mod tracking {
    use std::cell::UnsafeCell;

    use super::ID;

    thread_local! {
        pub static EVENTS: UnsafeCell<Vec<Event>> = const { UnsafeCell::new(vec![]) };
    }

    #[derive(Debug, Clone, Copy)]
    pub enum Event {
        Point { id: ID, level: usize },
        EdgeHorizontal { from: ID, to: ID, level: usize },
        EdgeDown { from: ID, upper_level: usize },
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

/*



fn insert_first(hnsw: &mut Hnsw, q: ID, mL: f32) {
     let l = pick_level(mL);
     // insert point at all levels
}


fn insert(
    hnsw: &mut Hnsw,
    q: ID,             id (idx) of pt
    M: usize,               number of establishedconnections M ??? , I think this means, how many neighbors to search for and compare on each layer????
    Mmax: usize,            maximum number of connections for each element per layer above layer 0
    Mmax_0: usize           maximum number of connections for each element on layer 0
    efConstruction: ??,
    mL: f32                  normalization factor for level generation
)   {
    let mut W : [ID]  = [];
    let mut ep : [ID] = [get_entry(hnsw)]  // entry pt for hnsw
    let L =   // level of ep
    let l = pick_level(mL);

    for lc in (l+1..=L).rev() {
        W = search_layer(hnsw, q, ep, ef = 1, lc)
        ep = [get_nearest_element(W, q)]
    }


    for lc in (0..=min(l,L)).rev() {
        let Mmax = if lc == 0 {  Mmax_0 } else { Mmax }
        W : [ID]  = search_layer(hnsw, q, ep, efConstruction, lc);
        neighbors = select_neighbors(hnsw, q, W, M, lc);

        // add bidirectional connections from neighbors to q at layer lc:

        for e in neighbors {
            // shrink connections if needed
            e_conn: [ID] = neighbourhood(hnsw, e, lc)
            if e_conn.len() > Mmax {
                // shrink connections of e:
                eNewConn = select_neighbors(hnsw, e, eConn, Mmax, lc);
                set_connections(hnsw, e, lc, eNewConn);
            }
        }

        ep = W
    }

    if l > L
        // set enter point for hnsw to q
        set_new_enter_point(hnsw, q, l)
}




fn set_new_enter_point(hnsw: &mut Hnsw, q: ID, l: usize) {
    // insert layers such that layer l can exist.
}

fn get_connections(hnsw: &Hnsw, e: ID, lc: usize) -> [ID] {




}

fn set_connections(hnsw: &mut Hnsw, e: ID, lc: usize, new_connections: [ID]) {


}

fn get_nearest_element(W: &[ID], q: &[f32]) -> ID {


}


fn select_neighbors(
    hnsw: &Hnsw,
    W: [ID],
    M: ??,
    lc: usize,        // layer we are looking at
) -> [ID]  {


}

fn search_layer(
    hnsw: &Hnsw,
    q: ID,
    ep: [ID]     // entry points
    ef: usize,   // number of elements to return
    lc: usize    // layer we are looking at
) -> ?? {
    let visited   : HashSet<ID>   =   ep; // visited elements
    let frontier                  =   ep; // candidates
    let found                     =   ep; // dyncamic list of found neighbors
    while candidates.len() > 0{
        c = extract nearest element from frontier to q
        f = get furthest element from found to q
        if distance(c, q) > distance(f,q){
            break; // all elements in W are evaluated
        }
        for e in get_connections(hnsw, c, lc) {
            if !visited.contains(e){
                visited.insert(e);
                f = get furthest element from found to q
                if distance(e, q) < distance (f,q)     || found.len()  <   ef{
                    frontier.push(e);
                    found.push(e);
                    if frontier.len() > ef {
                        // remove the furthest element from found to q.
                    }
                }
            }
        }
    }
    return found;
}


fn pick_level(mL: f32) -> usize {
    let r : f32 = rng.gen(); // 0 to 1
    let l = -ln(r)*mL;
    return floor(l);
}

fn top_level(hnsw: &Hnsw) -> usize  {
    hnsw.layers.len() - 1
}


fn get_entry(hnsw: &Hnsw) -> ID {
    self.layers.last().unwrap()[0]
}



*/