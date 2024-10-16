// src/mechanisms/mod.rs

/// Differential Privacy Mechanisms.
///
/// This module contains various mechanisms to ensure differential privacy
/// by adding noise or selecting outputs based on privacy-preserving algorithms.
pub mod laplace;

/// Expose the Laplace Mechanism for external use.
pub use laplace::laplace_mechanism;