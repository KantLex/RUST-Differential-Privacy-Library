// src/prelude.rs

//! Prelude module for easy imports.
//!
//! Import everything you need for the simplified API with a single statement:
//!
//! ```rust
//! use differential_privacy::prelude::*;
//!
//! // Now you can use simplified functions directly
//! let private_count = add_noise(100.0, PrivacyLevel::Medium);
//! ```
//!
//! This module re-exports the most commonly used items for beginners.

// Re-export everything from the simple module
pub use crate::simple::{
    // Privacy levels
    PrivacyLevel,

    // Simple noise functions
    add_noise,
    add_noise_with_sensitivity,

    // Simple aggregations
    private_sum_simple,
    private_mean_simple,
    private_count_simple,

    // Selection functions
    private_select,
    private_argmax,

    // Builder pattern
    PrivateQuery,

    // Utility functions
    estimate_noise,
    suggest_privacy_level,
};

// Also export the full API for users who want to gradually learn more
pub use crate::privacy_accounting::PrivacyAccountant;
