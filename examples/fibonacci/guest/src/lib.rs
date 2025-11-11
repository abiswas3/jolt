#![cfg_attr(feature = "guest", no_std)]
use jolt::{end_cycle_tracking, start_cycle_tracking};

#[jolt::provable(memory_size = 10240, max_trace_length = 65536)]
fn fib(_n: u32) -> u128 {
    let a: u128 = 0;
    let mut b: u128 = 1;

    start_cycle_tracking("fib_loop"); // Use `start_cycle_tracking("{name}")` to start a cycle span
    b = a + b;
    end_cycle_tracking("fib_loop"); // Use `end_cycle_tracking("{name}")` to end a cycle span
    b
}
