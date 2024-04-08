use crate::Float;

struct L1Distance;
struct L2SqDistance;

trait Distance {
    fn distance<const D: usize>(a: &[Float; D], b: &[Float; D]) -> Float;
}

impl Distance for L2SqDistance {
    fn distance<const D: usize>(a: &[Float; D], b: &[Float; D]) -> Float {
        let oct_loops: usize = D / 8;
        let remainder_lopps: usize = D % 8;

        let mut sum: Float = 0.0;

        todo!()
    }
}
