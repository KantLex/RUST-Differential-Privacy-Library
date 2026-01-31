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
//! ## Quick Start (Simplified API)
//!
//! For beginners, use the simplified API which requires minimal configuration:
//!
//! ```rust
//! use differential_privacy::prelude::*;
//!
//! // Add noise to a count with medium privacy
//! let private_count = add_noise(100.0, PrivacyLevel::Medium);
//! println!("Private count: {}", private_count);
//!
//! // Compute private statistics
//! let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
//! let private_avg = private_mean_simple(&values, 0.0, 100.0, PrivacyLevel::High);
//! println!("Private average: {:.1}", private_avg);
//! ```
//!
//! ## Full API
//!
//! For advanced users who need privacy accounting and composition:
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
//! - [`prelude`]: Easy imports for the simplified API (beginners start here)
//! - [`simple`]: Simplified functions that don't require privacy accounting
//! - [`privacy_accounting`]: Privacy budget tracking and composition
//! - [`mechanisms`]: Core DP mechanisms for adding noise
//! - [`aggregations`]: Private aggregation functions for arrays

/// Simplified API for Beginners
///
/// Easy-to-use functions that don't require understanding all the details
/// of differential privacy. Perfect for getting started quickly.
pub mod simple;

/// Prelude - Easy Imports
///
/// Import everything you need for the simplified API with a single `use` statement:
/// ```rust
/// use differential_privacy::prelude::*;
/// ```
pub mod prelude;

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
