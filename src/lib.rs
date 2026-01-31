// src/lib.rs

//! # Differential Privacy Library
//!
//! A comprehensive Rust library for implementing differential privacy in data analysis.
//!
//! ## Features
//!
//! - **Mechanisms**: Laplace, Gaussian, Exponential, Report Noisy Max
//! - **Aggregations**: Private sum, mean, count, variance, histogram
//! - **Privacy Accounting**: Basic, Advanced, and RDP composition
//! - **Budget Enforcement**: Automatic budget checking and enforcement
//!
//! ## Quick Start
//!
//! ```rust
//! use differential_privacy::mechanisms::laplace_mechanism;
//! use differential_privacy::privacy_accounting::PrivacyAccountant;
//!
//! let mut accountant = PrivacyAccountant::new();
//! let noisy_value = laplace_mechanism(100.0, 1.0, 0.5, &mut accountant);
//! println!("Noisy value: {}", noisy_value);
//! ```
//!
//! ## Modules
//!
//! - [`privacy_accounting`]: Privacy budget tracking and composition
//! - [`mechanisms`]: Core DP mechanisms for adding noise
//! - [`aggregations`]: Private aggregation functions for arrays

/// Privacy Accounting Module
///
/// Tracks cumulative privacy loss with support for multiple composition methods
/// including basic, advanced (Dwork-Rothblum-Vadhan), and Rényi DP composition.
pub mod privacy_accounting;

/// Differential Privacy Mechanisms
///
/// Core mechanisms for adding noise to queries:
/// - Laplace Mechanism (ε-DP)
/// - Gaussian Mechanism ((ε,δ)-DP)
/// - Exponential Mechanism (ε-DP for discrete outputs)
/// - Report Noisy Max (ε-DP for argmax queries)
pub mod mechanisms;

/// Private Aggregation Functions
///
/// Differentially private versions of common statistical aggregations:
/// - `private_sum`: Sum with Laplace noise
/// - `private_mean`: Mean with noise added to sum and count
/// - `private_count`: Count with Laplace noise
/// - `private_variance`: Variance with bounded sensitivity
/// - `private_histogram`: Histogram with per-bin noise
/// - Vector noise functions for array operations
pub mod aggregations;
