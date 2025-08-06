use std::sync::atomic::{AtomicUsize, Ordering};

pub static MULT_COUNT: AtomicUsize = AtomicUsize::new(0);

pub fn reset_mult_count() {
    MULT_COUNT.store(0, Ordering::Relaxed);
}

pub fn get_mult_count() -> usize {
    MULT_COUNT.load(Ordering::Relaxed)
}
