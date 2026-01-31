// src/simple.rs

//! Simplified API for beginners.
//!
//! This module provides easy-to-use functions that don't require understanding
//! all the details of differential privacy. Perfect for getting started quickly.
//!
//! # Quick Start
//!
//! ```rust
//! use differential_privacy::simple::*;
//!
//! // Add noise to a count with medium privacy
//! let private_count = add_noise(100.0, PrivacyLevel::Medium);
//!
//! // Or use a one-liner for common operations
//! let private_sum = private_sum_simple(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.0, 10.0, PrivacyLevel::High);
//! ```

use rand::Rng;

/// Privacy level presets for beginners.
///
/// These presets provide sensible defaults for different use cases:
///
/// - **Low**: ε=1.0 - Weak privacy, high accuracy. For non-sensitive data.
/// - **Medium**: ε=0.5 - Balanced privacy/accuracy. Good default choice.
/// - **High**: ε=0.1 - Strong privacy, moderate noise. For sensitive data.
/// - **VeryHigh**: ε=0.01 - Very strong privacy, significant noise. For highly sensitive data.
///
/// # Example
///
/// ```rust
/// use differential_privacy::simple::{add_noise, PrivacyLevel};
///
/// let value = 100.0;
/// let private_value = add_noise(value, PrivacyLevel::Medium);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrivacyLevel {
    /// Weak privacy (ε=1.0). High accuracy, suitable for non-sensitive data.
    Low,
    /// Balanced privacy (ε=0.5). Good default for most use cases.
    Medium,
    /// Strong privacy (ε=0.1). For sensitive data.
    High,
    /// Very strong privacy (ε=0.01). For highly sensitive data.
    VeryHigh,
    /// Custom epsilon value.
    Custom(f64),
}

impl PrivacyLevel {
    /// Get the epsilon value for this privacy level.
    pub fn epsilon(&self) -> f64 {
        match self {
            PrivacyLevel::Low => 1.0,
            PrivacyLevel::Medium => 0.5,
            PrivacyLevel::High => 0.1,
            PrivacyLevel::VeryHigh => 0.01,
            PrivacyLevel::Custom(eps) => *eps,
        }
    }

    /// Get a human-readable description of this privacy level.
    pub fn description(&self) -> &'static str {
        match self {
            PrivacyLevel::Low => "Low privacy (ε=1.0): Weak protection, high accuracy",
            PrivacyLevel::Medium => "Medium privacy (ε=0.5): Balanced protection and accuracy",
            PrivacyLevel::High => "High privacy (ε=0.1): Strong protection, moderate noise",
            PrivacyLevel::VeryHigh => "Very high privacy (ε=0.01): Maximum protection, significant noise",
            PrivacyLevel::Custom(eps) => {
                if *eps >= 1.0 {
                    "Custom: Weak privacy"
                } else if *eps >= 0.5 {
                    "Custom: Moderate privacy"
                } else if *eps >= 0.1 {
                    "Custom: Strong privacy"
                } else {
                    "Custom: Very strong privacy"
                }
            }
        }
    }
}

// ============================================================================
// Simple standalone functions (no accountant needed)
// ============================================================================

/// Samples noise from a Laplace distribution.
fn sample_laplace(scale: f64) -> f64 {
    let uniform: f64 = rand::thread_rng().gen::<f64>() - 0.5;
    -(scale) * uniform.signum() * (1.0 - 2.0 * uniform.abs()).ln()
}

/// Add Laplace noise to a value.
///
/// This is the simplest way to make a value private. Assumes sensitivity of 1.0,
/// which is correct for counting queries.
///
/// # Arguments
///
/// * `value` - The value to make private.
/// * `privacy` - The desired privacy level.
///
/// # Example
///
/// ```rust
/// use differential_privacy::simple::{add_noise, PrivacyLevel};
///
/// let count = 42.0;
/// let private_count = add_noise(count, PrivacyLevel::Medium);
/// println!("Private count: {}", private_count);
/// ```
pub fn add_noise(value: f64, privacy: PrivacyLevel) -> f64 {
    add_noise_with_sensitivity(value, 1.0, privacy)
}

/// Add Laplace noise with custom sensitivity.
///
/// Use this when your query has sensitivity other than 1.0.
///
/// # Arguments
///
/// * `value` - The value to make private.
/// * `sensitivity` - How much the query can change when one record changes.
/// * `privacy` - The desired privacy level.
///
/// # Example
///
/// ```rust
/// use differential_privacy::simple::{add_noise_with_sensitivity, PrivacyLevel};
///
/// // Sum query where values are bounded [0, 100]
/// let sum = 5000.0;
/// let private_sum = add_noise_with_sensitivity(sum, 100.0, PrivacyLevel::High);
/// ```
pub fn add_noise_with_sensitivity(value: f64, sensitivity: f64, privacy: PrivacyLevel) -> f64 {
    let epsilon = privacy.epsilon();
    let scale = sensitivity / epsilon;
    value + sample_laplace(scale)
}

/// Compute a private sum of values.
///
/// Values are automatically clipped to the specified bounds.
///
/// # Arguments
///
/// * `values` - The values to sum.
/// * `lower` - Lower bound for clipping.
/// * `upper` - Upper bound for clipping.
/// * `privacy` - The desired privacy level.
///
/// # Example
///
/// ```rust
/// use differential_privacy::simple::{private_sum_simple, PrivacyLevel};
///
/// let salaries = vec![50000.0, 60000.0, 55000.0, 70000.0];
/// let private_total = private_sum_simple(&salaries, 0.0, 100000.0, PrivacyLevel::Medium);
/// println!("Private total: {:.0}", private_total);
/// ```
pub fn private_sum_simple(values: &[f64], lower: f64, upper: f64, privacy: PrivacyLevel) -> f64 {
    let clipped_sum: f64 = values
        .iter()
        .map(|&v| v.clamp(lower, upper))
        .sum();

    let sensitivity = upper - lower;
    add_noise_with_sensitivity(clipped_sum, sensitivity, privacy)
}

/// Compute a private mean of values.
///
/// Values are automatically clipped to the specified bounds.
///
/// # Arguments
///
/// * `values` - The values to average.
/// * `lower` - Lower bound for clipping.
/// * `upper` - Upper bound for clipping.
/// * `privacy` - The desired privacy level.
///
/// # Example
///
/// ```rust
/// use differential_privacy::simple::{private_mean_simple, PrivacyLevel};
///
/// let ages = vec![25.0, 30.0, 35.0, 40.0, 45.0];
/// let private_avg = private_mean_simple(&ages, 0.0, 120.0, PrivacyLevel::Medium);
/// println!("Private average age: {:.1}", private_avg);
/// ```
pub fn private_mean_simple(values: &[f64], lower: f64, upper: f64, privacy: PrivacyLevel) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let n = values.len() as f64;
    let clipped_sum: f64 = values
        .iter()
        .map(|&v| v.clamp(lower, upper))
        .sum();

    let true_mean = clipped_sum / n;
    let sensitivity = (upper - lower) / n;
    add_noise_with_sensitivity(true_mean, sensitivity, privacy)
}

/// Compute a private count.
///
/// # Arguments
///
/// * `count` - The true count.
/// * `privacy` - The desired privacy level.
///
/// # Example
///
/// ```rust
/// use differential_privacy::simple::{private_count_simple, PrivacyLevel};
///
/// let num_users = 1000;
/// let private_num = private_count_simple(num_users, PrivacyLevel::Medium);
/// println!("Private user count: {}", private_num.round() as i64);
/// ```
pub fn private_count_simple(count: usize, privacy: PrivacyLevel) -> f64 {
    add_noise(count as f64, privacy)
}

/// Privately select from candidates based on scores.
///
/// Higher scores are more likely to be selected, but the selection is randomized
/// to preserve privacy.
///
/// # Arguments
///
/// * `candidates` - Names or labels for each candidate.
/// * `scores` - Utility score for each candidate (higher = better).
/// * `privacy` - The desired privacy level.
///
/// # Returns
///
/// The index of the selected candidate.
///
/// # Example
///
/// ```rust
/// use differential_privacy::simple::{private_select, PrivacyLevel};
///
/// let options = vec!["Option A", "Option B", "Option C"];
/// let scores = vec![10.0, 15.0, 12.0];
/// let selected = private_select(&options, &scores, PrivacyLevel::Medium);
/// println!("Selected: {}", options[selected]);
/// ```
pub fn private_select<T>(candidates: &[T], scores: &[f64], privacy: PrivacyLevel) -> usize {
    assert!(!candidates.is_empty(), "Candidates cannot be empty");
    assert_eq!(candidates.len(), scores.len(), "Candidates and scores must have same length");

    let epsilon = privacy.epsilon();
    let sensitivity = 1.0; // Assume sensitivity 1 for simplicity

    // Compute selection probabilities using exponential mechanism
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scores
        .iter()
        .map(|&s| ((epsilon * (s - max_score)) / (2.0 * sensitivity)).exp())
        .collect();

    let total: f64 = weights.iter().sum();
    let normalized: Vec<f64> = weights.iter().map(|w| w / total).collect();

    // Sample from the distribution
    let mut rng = rand::thread_rng();
    let r: f64 = rng.gen();
    let mut cumsum = 0.0;

    for (i, &p) in normalized.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }

    candidates.len() - 1
}

/// Privately find the index with the maximum count.
///
/// # Arguments
///
/// * `counts` - The counts for each category.
/// * `privacy` - The desired privacy level.
///
/// # Returns
///
/// The index of the (approximately) maximum count.
///
/// # Example
///
/// ```rust
/// use differential_privacy::simple::{private_argmax, PrivacyLevel};
///
/// let votes = vec![150.0, 200.0, 175.0];  // Votes for 3 candidates
/// let winner = private_argmax(&votes, PrivacyLevel::Medium);
/// println!("Winner is candidate {}", winner);
/// ```
pub fn private_argmax(counts: &[f64], privacy: PrivacyLevel) -> usize {
    assert!(!counts.is_empty(), "Counts cannot be empty");

    let epsilon = privacy.epsilon();
    let sensitivity = 1.0;
    let scale = sensitivity / epsilon;

    // Add noise to each count and find argmax
    counts
        .iter()
        .enumerate()
        .map(|(i, &c)| (i, c + sample_laplace(scale)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

// ============================================================================
// Builder pattern for more complex queries
// ============================================================================

/// Builder for constructing private queries with more control.
///
/// # Example
///
/// ```rust
/// use differential_privacy::simple::PrivateQuery;
///
/// let result = PrivateQuery::new(100.0)
///     .sensitivity(10.0)
///     .epsilon(0.5)
///     .release();
///
/// println!("Private result: {}", result);
/// ```
#[derive(Debug, Clone)]
pub struct PrivateQuery {
    value: f64,
    sensitivity: f64,
    epsilon: f64,
}

impl PrivateQuery {
    /// Create a new private query for a value.
    pub fn new(value: f64) -> Self {
        Self {
            value,
            sensitivity: 1.0,
            epsilon: 0.5, // Default to medium privacy
        }
    }

    /// Set the sensitivity of the query.
    pub fn sensitivity(mut self, sensitivity: f64) -> Self {
        self.sensitivity = sensitivity;
        self
    }

    /// Set the epsilon (privacy budget) directly.
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Use a privacy level preset.
    pub fn privacy(mut self, level: PrivacyLevel) -> Self {
        self.epsilon = level.epsilon();
        self
    }

    /// Release the private value with Laplace noise.
    pub fn release(self) -> f64 {
        let scale = self.sensitivity / self.epsilon;
        self.value + sample_laplace(scale)
    }

    /// Get the expected noise magnitude (scale parameter).
    pub fn expected_noise_scale(&self) -> f64 {
        self.sensitivity / self.epsilon
    }
}

// ============================================================================
// Utility functions
// ============================================================================

/// Estimate how much noise will be added for a given configuration.
///
/// Returns the scale parameter of the Laplace distribution.
/// About 63% of noise samples fall within ±scale, and 95% within ±3*scale.
///
/// # Example
///
/// ```rust
/// use differential_privacy::simple::{estimate_noise, PrivacyLevel};
///
/// let scale = estimate_noise(100.0, PrivacyLevel::Medium);
/// println!("Expected noise scale: ±{:.1}", scale);
/// println!("95% of results will be within ±{:.1} of true value", 3.0 * scale);
/// ```
pub fn estimate_noise(sensitivity: f64, privacy: PrivacyLevel) -> f64 {
    sensitivity / privacy.epsilon()
}

/// Suggest which privacy level to use based on dataset size.
///
/// Larger datasets can tolerate stronger privacy because noise has less
/// relative impact on aggregate statistics.
///
/// # Example
///
/// ```rust
/// use differential_privacy::simple::suggest_privacy_level;
///
/// let level = suggest_privacy_level(10000);
/// println!("Suggested privacy level: {:?}", level);
/// ```
pub fn suggest_privacy_level(dataset_size: usize) -> PrivacyLevel {
    match dataset_size {
        0..=100 => PrivacyLevel::Low,
        101..=1000 => PrivacyLevel::Medium,
        1001..=10000 => PrivacyLevel::High,
        _ => PrivacyLevel::VeryHigh,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_privacy_level_epsilon() {
        assert_eq!(PrivacyLevel::Low.epsilon(), 1.0);
        assert_eq!(PrivacyLevel::Medium.epsilon(), 0.5);
        assert_eq!(PrivacyLevel::High.epsilon(), 0.1);
        assert_eq!(PrivacyLevel::VeryHigh.epsilon(), 0.01);
        assert_eq!(PrivacyLevel::Custom(0.25).epsilon(), 0.25);
    }

    #[test]
    fn test_add_noise() {
        let value = 100.0;
        let noisy = add_noise(value, PrivacyLevel::Medium);
        // Just check it produces a value (noise is random)
        assert!(noisy.is_finite());
    }

    #[test]
    fn test_add_noise_with_sensitivity() {
        let value = 100.0;
        let noisy = add_noise_with_sensitivity(value, 10.0, PrivacyLevel::High);
        assert!(noisy.is_finite());
    }

    #[test]
    fn test_private_sum_simple() {
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let sum = private_sum_simple(&values, 0.0, 100.0, PrivacyLevel::Low);
        // True sum is 150, noise should be bounded
        assert!(sum > 0.0 && sum < 500.0);
    }

    #[test]
    fn test_private_mean_simple() {
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mean = private_mean_simple(&values, 0.0, 100.0, PrivacyLevel::Low);
        // True mean is 30, check it's reasonable
        assert!(mean.is_finite());
    }

    #[test]
    fn test_private_mean_empty() {
        let values: Vec<f64> = vec![];
        let mean = private_mean_simple(&values, 0.0, 100.0, PrivacyLevel::Medium);
        assert_eq!(mean, 0.0);
    }

    #[test]
    fn test_private_count_simple() {
        let count = private_count_simple(100, PrivacyLevel::Medium);
        assert!(count.is_finite());
    }

    #[test]
    fn test_private_select() {
        let candidates = vec!["A", "B", "C"];
        let scores = vec![1.0, 100.0, 1.0];  // B is much better

        // With low privacy (high epsilon), should usually select B
        let mut b_count = 0;
        for _ in 0..100 {
            if private_select(&candidates, &scores, PrivacyLevel::Low) == 1 {
                b_count += 1;
            }
        }
        // B should be selected most of the time
        assert!(b_count > 50);
    }

    #[test]
    fn test_private_argmax() {
        let counts = vec![10.0, 100.0, 20.0];  // Index 1 is max

        // With low privacy, should usually find index 1
        let mut correct = 0;
        for _ in 0..100 {
            if private_argmax(&counts, PrivacyLevel::Low) == 1 {
                correct += 1;
            }
        }
        assert!(correct > 50);
    }

    #[test]
    fn test_private_query_builder() {
        let result = PrivateQuery::new(100.0)
            .sensitivity(10.0)
            .epsilon(0.5)
            .release();
        assert!(result.is_finite());
    }

    #[test]
    fn test_private_query_with_privacy_level() {
        let result = PrivateQuery::new(100.0)
            .sensitivity(5.0)
            .privacy(PrivacyLevel::High)
            .release();
        assert!(result.is_finite());
    }

    #[test]
    fn test_estimate_noise() {
        let scale = estimate_noise(1.0, PrivacyLevel::Medium);
        assert_eq!(scale, 2.0);  // 1.0 / 0.5 = 2.0
    }

    #[test]
    fn test_suggest_privacy_level() {
        assert_eq!(suggest_privacy_level(50), PrivacyLevel::Low);
        assert_eq!(suggest_privacy_level(500), PrivacyLevel::Medium);
        assert_eq!(suggest_privacy_level(5000), PrivacyLevel::High);
        assert_eq!(suggest_privacy_level(50000), PrivacyLevel::VeryHigh);
    }
}
