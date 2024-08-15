use core::f32;
use std::sync::Mutex;

use rand::{thread_rng, Rng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use vecnn::{dataset::FlatDataSet, distance::Distance};

fn main() {
    let distance = Distance::L2;
    let distance = distance.to_fn();
    let n = 100;
    let n2 = 1000;

    #[derive(Debug, Clone)]
    struct Res {
        d: usize,
        mean: f32,
        var: f32,
        max_min_div_min: f32,
    }

    let results: Mutex<Vec<Res>> = Mutex::new(vec![]);

    (1..=1000).into_par_iter().for_each(|d| {
        let mut distances: Vec<f32> = vec![];

        let mut a: Vec<f32> = vec![0.0; d];
        let mut b: Vec<f32> = vec![0.0; d];

        let mut mean_2: f32 = 0.0;
        let mut var_2: f32 = 0.0;
        let mut max_min_div_min_2: f32 = 0.0;
        for _ in 0..n2 {
            for _ in 0..n {
                random_fill(&mut a);
                random_fill(&mut b);
                let dist = distance(&a, &b);
                distances.push(dist);
            }

            let mut mean: f32 = 0.0;
            let mut min: f32 = f32::MAX;
            let mut max: f32 = f32::MIN;

            for &d in distances.iter() {
                mean += d;
                if d > max {
                    max = d;
                }
                if d < min {
                    min = d;
                }
            }
            mean /= n as f32;

            let mut var: f32 = 0.0;
            for &d in distances.iter() {
                var += (d - mean) * (d - mean)
            }
            var /= n as f32;

            var_2 += var;
            mean_2 += mean;
            max_min_div_min_2 += (max - min) / min;
        }
        mean_2 /= n2 as f32;
        var_2 /= n2 as f32;
        max_min_div_min_2 /= n2 as f32;

        results.lock().unwrap().push(Res {
            d,
            mean: mean_2,
            var: var_2,
            max_min_div_min: max_min_div_min_2,
        });
    });

    let mut results = results.lock().unwrap().clone();
    results.sort_by_key(|e| e.d);

    println!("d,mean,var,(max-min)/min");
    for r in results {
        let Res {
            d,
            mean,
            var,
            max_min_div_min,
        } = r;
        println!("{d},{mean},{var},{max_min_div_min}",);
    }
}

fn random_fill(a: &mut [f32]) {
    let mut rng = thread_rng();
    for e in a.iter_mut() {
        *e = rng.gen();
    }
}
