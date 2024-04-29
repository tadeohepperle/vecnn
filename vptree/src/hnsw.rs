use std::{
    collections::{BinaryHeap, HashSet},
    sync::Arc,
};

use arrayvec::ArrayVec;
use rand::{thread_rng, Rng};
use rand_chacha::ChaChaRng;

use crate::{
    dataset::DatasetT,
    distance::{DistanceT, SquaredDiffSum},
};

#[derive(Debug, Clone, Copy)]
pub struct Params {
    /// normalization factor for level generation
    /// Influences the chance of at which level a point is interted.
    level_norm_param: f32,
    ef_construction: usize,
    m_max: usize,
    m_max_0: usize,
}

pub struct Hnsw {
    params: Params,
    data: Arc<dyn DatasetT>,
    layers: Vec<Layer>,
}

struct Layer {
    entries: Vec<LayerEntry>,
}

type ID = u32;
const M: usize = 40;

struct LayerEntry {
    id: ID,
    /// pos where we can find this entry at a lower level.
    /// insignificat on level 0, just set to u32::MAX.
    lower_level_idx: u32,
    neighbors: ArrayVec<u32, M>, // the u32 stores the index in the layer
}
impl LayerEntry {
    fn new(id: u32, lower_level_idx: u32) -> LayerEntry {
        LayerEntry {
            id,
            lower_level_idx,
            neighbors: ArrayVec::new(),
        }
    }
}

impl Hnsw {
    pub fn new(data: Arc<dyn DatasetT>, params: Params) -> Self {
        new_hnsw(data, params)
    }
}

fn new_hnsw(data: Arc<dyn DatasetT>, params: Params) -> Hnsw {
    let mut hnsw = Hnsw {
        params,
        data,
        layers: vec![],
    };

    let len = hnsw.data.len();
    if len == 0 {
        return hnsw;
    }

    // insert a first layer with a first entry
    let mut entries = Vec::with_capacity(len);
    entries.push(LayerEntry {
        id: 0,
        lower_level_idx: u32::MAX,
        neighbors: ArrayVec::new(),
    });
    hnsw.layers.push(Layer { entries });

    // insert the rest of the points one by one
    for id in 1..len as u32 {
        insert(&mut hnsw, id);
    }

    hnsw
}

struct Idk;

fn insert(hnsw: &mut Hnsw, q: ID) {
    let Params {
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
    let mut levels_added: usize = 0;
    let mut lower_level_idx: u32 = u32::MAX;
    for l in 0..=insert_l {
        let entry = LayerEntry::new(q, lower_level_idx);
        if let Some(layer) = hnsw.layers.get_mut(l) {
            lower_level_idx = layer.entries.len() as u32;
            layer.entries.push(entry);
        } else {
            let layer = Layer {
                entries: vec![entry],
            };
            hnsw.layers.push(layer);
            lower_level_idx = 0;
            levels_added += 1;
        }
    }

    // /////////////////////////////////////////////////////////////////////////////
    // Phase 1: find the idx of the entry point at level `insert_l`
    // /////////////////////////////////////////////////////////////////////////////

    // this loop only runs, if insert_l < top_l, otherwise, the entry point is just the first point of the highest level.
    let ep_l = top_l.min(insert_l);
    let mut ep_idx_at_ep_l = 0;
    for l in (insert_l + 1..=top_l).rev() {
        let res = closest_point_in_layer(&hnsw.layers[l], q_data, ep_idx_at_ep_l);
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
        );
        select_neighbors(
            layer,
            q_data,
            search_buffers.found.as_slice(),
            &mut neighbors_out,
        );
        // add bidirectional connections from neighbors to q at layer l:
        let idx_of_q_in_l = layer.entries.len() as u32 - 1;
        for n in neighbors_out.iter() {
            layer.entries[n.idx_in_layer].neighbors.push(idx_of_q_in_l);
            layer.entries[n.idx_in_layer].neighbors.push(idx_of_q_in_l);
        }
        for n in neighbors_out.iter() {}
    }

    // add additional layers if necessary:
    // if insert_l > top_l {
    //     let mut lower_level_i = hnsw.layers[top_l].entries.len() as u32;
    //     for _ in 0..(insert_l - top_l) {
    //         hnsw.layers.push(Layer {
    //             entries: vec![LayerEntry {
    //                 id,
    //                 lower_level_idx,
    //                 neighbors: ArrayVec::new(),
    //             }],
    //         });
    //         lower_level_i = 0;
    //     }
    // }

    // for l in (0..=insert_l).rev() {}

    todo!()

    // for lc in (l+1..=L).rev() {
    //     W = search_layer(hnsw, q, ep, ef = 1, lc)
    //     ep = [get_nearest_element(W, q)]
    // }

    // for lc in (0..=min(l,L)).rev() {
    //     let Mmax = if lc == 0 {  Mmax_0 } else { Mmax }
    //     W : [ID]  = search_layer(hnsw, q, ep, efConstruction, lc);
    //     neighbors = select_neighbors(hnsw, q, W, M, lc);

    //     // add bidirectional connections from neighbors to q at layer lc:

    //     for e in neighbors {
    //         // shrink connections if needed
    //         e_conn: [ID] = neighbourhood(hnsw, e, lc)
    //         if e_conn.len() > Mmax {
    //             // shrink connections of e:
    //             eNewConn = select_neighbors(hnsw, e, eConn, Mmax, lc);
    //             set_connections(hnsw, e, lc, eNewConn);
    //         }
    //     }

    //     ep = W
    // }

    // if l > L
    //     // set enter point for hnsw to q
    //     set_new_enter_point(hnsw, q, l)
}

// fn insert_phase_1()

fn pick_level(level_norm_param: f32) -> usize {
    let f = thread_rng().gen::<f32>();
    (-f.ln() * level_norm_param).floor() as usize
}

#[test]
fn testlevel() {
    for i in 0..100 {
        println!("{}", pick_level(10.0))
    }
}

struct HnswEp {
    top_l: usize,
    ep: PtInLayer,
}

struct PtInLayer {
    idx_in_layer: usize,
    id: ID,
}

/// panics if no point present. returns the points id,
fn get_entry(hnsw: &Hnsw) -> HnswEp {
    let top_l = hnsw.layers.len() - 1;
    let idx_in_layer = 0;
    let id = hnsw.layers[top_l].entries[idx_in_layer].id;
    HnswEp {
        top_l,
        ep: PtInLayer { idx_in_layer, id },
    }
}

#[derive(Debug, Clone, Copy)]
struct SearchLayerRes {
    idx_in_layer: usize,
    idx_in_lower_layer: usize,
    id: ID,
    d_to_q: f32,
}

fn closest_point_in_layer(layer: &Layer, q_data: &[f32], ep_idx_in_layer: usize) -> SearchLayerRes {
    todo!()
}

struct SearchBuffers {
    visited_idxs: HashSet<usize>,
    /// we need to be able to extract the closest element from this.
    candidates: BinaryHeap<ClosestIsRoot>,
    /// we need to be able to extract the furthest element from this.
    found: BinaryHeap<FurthestIsRoot>,
}

impl SearchBuffers {
    fn new(capacity: usize) -> Self {
        SearchBuffers {
            visited_idxs: HashSet::with_capacity(capacity),
            candidates: BinaryHeap::with_capacity(capacity),
            found: BinaryHeap::with_capacity(capacity),
        }
    }

    fn initialize(&mut self, ep_idxs_in_layer: &[usize], idx_to_dist: impl Fn(usize) -> f32) {
        self.visited_idxs.clear();
        self.candidates.clear();
        self.found.clear();
        for idx in ep_idxs_in_layer {
            let d = idx_to_dist(*idx);
            self.visited_idxs.insert(*idx);
            self.candidates.push(ClosestIsRoot {
                idx_in_layer: *idx,
                dist: d,
            });
            self.found.push(FurthestIsRoot {
                idx_in_layer: *idx,
                dist: d,
            })
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FurthestIsRoot {
    idx_in_layer: usize,
    dist: f32, // distance
}

impl PartialEq for FurthestIsRoot {
    fn eq(&self, other: &Self) -> bool {
        self.idx_in_layer == other.idx_in_layer && self.dist == other.dist
    }
}
impl Eq for FurthestIsRoot {}
impl PartialOrd for FurthestIsRoot {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}
impl Ord for FurthestIsRoot {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.total_cmp(&other.dist)
    }
}

#[derive(Debug, Clone, Copy)]
struct ClosestIsRoot {
    idx_in_layer: usize,
    dist: f32, // distance
}

impl PartialEq for ClosestIsRoot {
    fn eq(&self, other: &Self) -> bool {
        self.idx_in_layer == other.idx_in_layer && self.dist == other.dist
    }
}
impl Eq for ClosestIsRoot {}
impl PartialOrd for ClosestIsRoot {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.dist.partial_cmp(&self.dist)
    }
}
impl Ord for ClosestIsRoot {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.dist.total_cmp(&self.dist)
    }
}

/// after doing this, out.found will contain the relevant found points.
fn closests_points_in_layer(
    layer: &Layer,
    data: &dyn DatasetT,
    q_data: &[f32],
    ep_idxs_in_layer: &[usize],
    ef: usize, // max number of found items
    out: &mut SearchBuffers,
) {
    let distance = SquaredDiffSum::distance;
    out.initialize(ep_idxs_in_layer, |idx| distance(data.get(idx), q_data));

    while out.candidates.len() > 0 {
        let c = out.candidates.pop().unwrap(); // remove closest element.
        let mut f = *out.found.peek().unwrap();
        if c.dist > f.dist {
            break; // all elements in found are evaluated (see paper).
        }
        let c_entry = &layer.entries[c.idx_in_layer];
        for idx_in_layer in c_entry.neighbors.iter() {
            let idx_in_layer = *idx_in_layer as usize;
            if out.visited_idxs.insert(idx_in_layer) {
                let n_entry = &layer.entries[idx_in_layer];
                let n_data = data.get(n_entry.id as usize);
                let dist = distance(q_data, n_data);
                f = *out.found.peek().unwrap();
                if dist < f.dist || out.found.len() < ef {
                    out.candidates.push(ClosestIsRoot { idx_in_layer, dist });

                    if out.found.len() < ef {
                        out.found.push(FurthestIsRoot { idx_in_layer, dist });
                    } else {
                        // compare dist to the currently furthest away,
                        // if further than this dist, kick it out and insert the new one instead.
                        if dist < f.dist {
                            out.found.pop().unwrap();
                            out.found.push(FurthestIsRoot { idx_in_layer, dist });
                        }
                    }
                }
            }
        }
    }
}

fn select_neighbors(
    layer: &Layer,
    q_data: &[f32],
    candidates: &[FurthestIsRoot],
    out: &mut Vec<SearchLayerRes>,
) {
    out.clear()
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
