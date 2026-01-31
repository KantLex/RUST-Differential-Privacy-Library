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
/// - **Sparse Vector Technique**: Answer many threshold queries with fixed privacy budget.
/// - **Privacy Amplification by Subsampling**: Stronger guarantees when using data subsamples.
///
/// ## Budget-Enforcing Mechanisms
///
/// The `budgeted` submodule provides versions of all mechanisms that automatically
/// check and enforce privacy budget limits before execution.

pub mod laplace;
pub mod gaussian;
pub mod exponential;
pub mod report_noisy_max;
pub mod budgeted;
pub mod sparse_vector;
pub mod subsampling;

/// Expose mechanism functions for external use.
pub use laplace::laplace_mechanism;
pub use gaussian::gaussian_mechanism;
pub use exponential::exponential_mechanism;
pub use report_noisy_max::{report_noisy_max, report_noisy_argmax};

/// Expose budget-enforcing mechanism functions.
pub use budgeted::{
    laplace_mechanism_budgeted,
    gaussian_mechanism_budgeted,
    exponential_mechanism_budgeted,
    report_noisy_max_budgeted,
    BudgetedAccountant,
    BudgetedMechanismError,
};

/// Expose Sparse Vector Technique.
pub use sparse_vector::{
    SparseVectorTechnique,
    NumericSparseVector,
    ThresholdResult,
    NumericThresholdResult,
    sparse_vector_find_first,
    sparse_vector_find_all,
};

/// Expose Privacy Amplification by Subsampling.
pub use subsampling::{
    amplify_epsilon_poisson,
    amplify_epsilon_uniform,
    amplify_epsilon_delta_poisson,
    compute_base_epsilon,
    poisson_subsample,
    poisson_subsample_indices,
    uniform_subsample,
    SubsampledMechanism,
    subsampled_laplace_sum,
    subsampled_laplace_mean,
    subsampling_noise_reduction,
};
