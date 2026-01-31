// src/mechanisms/mod.rs

/// Differential Privacy Mechanisms.
///
/// This module contains various mechanisms to ensure differential privacy
/// by adding noise or selecting outputs based on privacy-preserving algorithms.
///
/// ## Available Mechanisms
///
/// - **Laplace Mechanism**: Adds Laplace-distributed noise for ε-differential privacy.
/// - **Gaussian Mechanism**: Adds Gaussian-distributed noise for (ε, δ)-differential privacy.
/// - **Exponential Mechanism**: Privately selects an item from candidates based on utility scores.
/// - **Report Noisy Max**: Privately reports the index of the maximum value in a set of counts.

pub mod laplace;
pub mod gaussian;
pub mod exponential;
pub mod report_noisy_max;

/// Expose mechanism functions for external use.
pub use laplace::laplace_mechanism;
pub use gaussian::gaussian_mechanism;
pub use exponential::exponential_mechanism;
pub use report_noisy_max::{report_noisy_max, report_noisy_argmax};
