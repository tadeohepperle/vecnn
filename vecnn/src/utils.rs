use rand::{thread_rng, Rng};

use crate::{
    dataset::{DatasetT, FlatDataSet},
    distance::{DistanceT, SquaredDiffSum},
    hnsw::DistAnd,
    Float,
};
use std::{collections::BinaryHeap, sync::Arc};

pub fn linear_knn_search(data: &dyn DatasetT, q_data: &[f32], k: usize) -> Vec<DistAnd<u32>> {
    assert_eq!(q_data.len(), data.dims());
    // this stores the item with the greatest distance in the root (first element)
    let mut best: BinaryHeap<DistAnd<u32>> = BinaryHeap::new();

    let dist_fn = |i_data: &[f32]| SquaredDiffSum::distance(q_data, i_data);
    for id in 0..data.len() {
        let i_data = data.get(id);
        let dist = dist_fn(i_data);
        if best.len() < k {
            best.push(DistAnd { dist, i: id as u32 });
        } else {
            let worst = best.peek().unwrap();
            if worst.dist > dist {
                best.pop();
                best.push(DistAnd { dist, i: id as u32 });
            }
        }
    }

    let mut res = Vec::from(best);
    res.reverse();
    res
}

/// just draw numbers 0..X where X <=9 into the grid to test stuff:
pub fn simple_test_set() -> Arc<dyn DatasetT> {
    let str = "
      3
    4
                            7 
                   2
        9
    6                      0
              5
                     8  1   
    ";

    let mut pts: Vec<(usize, [f32; 2])> = vec![];
    let mut y = 0.0;
    for line in str.lines() {
        let mut x = 0.0;
        for ch in line.chars() {
            if let Some(idx) = ch.to_digit(10) {
                pts.push((idx as usize, [x, y]))
            }
            x += 1.0;
        }
        y += 2.0;
    }
    pts.sort_by(|a, b| a.0.cmp(&b.0));
    for (i, (e, _)) in pts.iter().enumerate() {
        assert_eq!(i, *e)
    }
    let pts: Vec<[f32; 2]> = pts.into_iter().map(|e| e.1).collect();
    Arc::new(pts)
}

pub fn random_data_set(len: usize, dims: usize) -> Arc<dyn DatasetT> {
    FlatDataSet::new_random(len, dims)
}

pub fn random_data_point<const DIMS: usize>() -> [Float; DIMS] {
    let mut rng = thread_rng();
    let mut p: [Float; DIMS] = [0.0; DIMS];
    for f in p.iter_mut() {
        *f = rng.gen();
    }
    p
}

#[cfg(test)]
mod test {
    use super::{linear_knn_search, simple_test_set};

    #[test]
    fn linear_knn() {
        let data = simple_test_set();

        let res = linear_knn_search(&*data, data.get(0), 4);
        let res: Vec<u32> = res.into_iter().map(|e| e.i).collect();
        assert_eq!(res, vec![0, 1, 7, 8]);
    }
}
