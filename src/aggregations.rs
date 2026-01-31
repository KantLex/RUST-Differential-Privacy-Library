// src/aggregations.rs

//! Private aggregation functions for differential privacy.
//!
//! This module provides differentially private versions of common aggregation
//! operations like sum, mean, count, and variance. These functions add calibrated
//! noise to ensure privacy while computing statistics over datasets.

use ndarray::{Array1, ArrayView1};
use crate::privacy_accounting::PrivacyAccountant;

/// Computes a differentially private sum of values.
///
/// Adds Laplace noise calibrated to the sensitivity of the sum operation.
/// The sensitivity depends on the range of possible values.
///
/// # Arguments
///
/// * `values` - The array of values to sum.
/// * `lower_bound` - The minimum possible value for any element.
/// * `upper_bound` - The maximum possible value for any element.
/// * `epsilon` - The privacy budget for this query.
/// * `accountant` - The privacy accountant to track privacy loss.
///
/// # Returns
///
/// The differentially private sum.
///
/// # Example
///
/// ```rust
/// use differential_privacy::aggregations::private_sum;
/// use differential_privacy::privacy_accounting::PrivacyAccountant;
/// use ndarray::array;
///
/// let values = array![10.0, 20.0, 30.0, 40.0];
/// let mut accountant = PrivacyAccountant::new();
///
/// let noisy_sum = private_sum(values.view(), 0.0, 100.0, 0.5, &mut accountant)
///     .expect("Invalid parameters");
/// println!("Private sum: {}", noisy_sum);
/// ```
pub fn private_sum(
    values: ArrayView1<f64>,
    lower_bound: f64,
    upper_bound: f64,
    epsilon: f64,
    accountant: &mut PrivacyAccountant,
) -> Result<f64, &'static str> {
    // Validate parameters
    if epsilon <= 0.0 {
        return Err("Epsilon must be positive.");
    }
    if lower_bound > upper_bound {
        return Err("Lower bound must be <= upper bound.");
    }

    // Clip values to the specified range
    let clipped: Array1<f64> = values.mapv(|v| v.clamp(lower_bound, upper_bound));

    // Compute the true sum
    let true_sum: f64 = clipped.sum();

    // Sensitivity of sum = upper_bound - lower_bound (one person can change the sum by this much)
    let sensitivity = upper_bound - lower_bound;

    // Add Laplace noise
    let noise = sample_laplace(sensitivity / epsilon);

    // Update privacy accountant
    accountant.update(epsilon, 0.0);

    Ok(true_sum + noise)
}

/// Computes a differentially private count of values matching a predicate.
///
/// # Arguments
///
/// * `values` - The array of values to count.
/// * `predicate` - A function that returns true for values to count.
/// * `epsilon` - The privacy budget for this query.
/// * `accountant` - The privacy accountant to track privacy loss.
///
/// # Returns
///
/// The differentially private count.
///
/// # Example
///
/// ```rust
/// use differential_privacy::aggregations::private_count;
/// use differential_privacy::privacy_accounting::PrivacyAccountant;
/// use ndarray::array;
///
/// let values = array![10.0, 20.0, 30.0, 40.0, 50.0];
/// let mut accountant = PrivacyAccountant::new();
///
/// // Count values greater than 25
/// let noisy_count = private_count(values.view(), |&x| x > 25.0, 0.5, &mut accountant)
///     .expect("Invalid parameters");
/// println!("Private count: {}", noisy_count);
/// ```
pub fn private_count<F>(
    values: ArrayView1<f64>,
    predicate: F,
    epsilon: f64,
    accountant: &mut PrivacyAccountant,
) -> Result<f64, &'static str>
where
    F: Fn(&f64) -> bool,
{
    if epsilon <= 0.0 {
        return Err("Epsilon must be positive.");
    }

    // Count matching values
    let true_count: f64 = values.iter().filter(|v| predicate(v)).count() as f64;

    // Sensitivity of count = 1 (one person can change the count by at most 1)
    let sensitivity = 1.0;

    // Add Laplace noise
    let noise = sample_laplace(sensitivity / epsilon);

    // Update privacy accountant
    accountant.update(epsilon, 0.0);

    Ok((true_count + noise).max(0.0)) // Count can't be negative
}

/// Computes a differentially private mean of values.
///
/// Uses the bounded mean approach: privately compute sum and count, then divide.
/// This provides (ε)-differential privacy.
///
/// # Arguments
///
/// * `values` - The array of values to average.
/// * `lower_bound` - The minimum possible value for any element.
/// * `upper_bound` - The maximum possible value for any element.
/// * `epsilon` - The privacy budget for this query (split between sum and count).
/// * `accountant` - The privacy accountant to track privacy loss.
///
/// # Returns
///
/// The differentially private mean.
///
/// # Example
///
/// ```rust
/// use differential_privacy::aggregations::private_mean;
/// use differential_privacy::privacy_accounting::PrivacyAccountant;
/// use ndarray::array;
///
/// let values = array![10.0, 20.0, 30.0, 40.0];
/// let mut accountant = PrivacyAccountant::new();
///
/// let noisy_mean = private_mean(values.view(), 0.0, 100.0, 0.5, &mut accountant)
///     .expect("Invalid parameters");
/// println!("Private mean: {}", noisy_mean);
/// ```
pub fn private_mean(
    values: ArrayView1<f64>,
    lower_bound: f64,
    upper_bound: f64,
    epsilon: f64,
    accountant: &mut PrivacyAccountant,
) -> Result<f64, &'static str> {
    if epsilon <= 0.0 {
        return Err("Epsilon must be positive.");
    }
    if lower_bound > upper_bound {
        return Err("Lower bound must be <= upper bound.");
    }
    if values.is_empty() {
        return Err("Values array must not be empty.");
    }

    // Split epsilon between sum and count
    let epsilon_sum = epsilon / 2.0;
    let epsilon_count = epsilon / 2.0;

    // Clip values
    let clipped: Array1<f64> = values.mapv(|v| v.clamp(lower_bound, upper_bound));

    // Private sum
    let true_sum: f64 = clipped.sum();
    let sum_sensitivity = upper_bound - lower_bound;
    let noisy_sum = true_sum + sample_laplace(sum_sensitivity / epsilon_sum);

    // Private count
    let true_count = values.len() as f64;
    let noisy_count = (true_count + sample_laplace(1.0 / epsilon_count)).max(1.0);

    // Update privacy accountant (total epsilon used)
    accountant.update(epsilon, 0.0);

    Ok(noisy_sum / noisy_count)
}

/// Computes a differentially private variance of values.
///
/// Uses a two-pass approach with clipping for bounded sensitivity.
///
/// # Arguments
///
/// * `values` - The array of values.
/// * `lower_bound` - The minimum possible value for any element.
/// * `upper_bound` - The maximum possible value for any element.
/// * `epsilon` - The privacy budget for this query.
/// * `accountant` - The privacy accountant to track privacy loss.
///
/// # Returns
///
/// The differentially private variance.
pub fn private_variance(
    values: ArrayView1<f64>,
    lower_bound: f64,
    upper_bound: f64,
    epsilon: f64,
    accountant: &mut PrivacyAccountant,
) -> Result<f64, &'static str> {
    if epsilon <= 0.0 {
        return Err("Epsilon must be positive.");
    }
    if lower_bound > upper_bound {
        return Err("Lower bound must be <= upper bound.");
    }
    if values.len() < 2 {
        return Err("Need at least 2 values for variance.");
    }

    // Split epsilon: sum, sum of squares, count
    let eps_per_query = epsilon / 3.0;

    // Clip values
    let clipped: Array1<f64> = values.mapv(|v| v.clamp(lower_bound, upper_bound));

    // Private sum
    let true_sum: f64 = clipped.sum();
    let sum_sensitivity = upper_bound - lower_bound;
    let noisy_sum = true_sum + sample_laplace(sum_sensitivity / eps_per_query);

    // Private sum of squares
    let true_sum_sq: f64 = clipped.mapv(|v| v * v).sum();
    let max_val = upper_bound.abs().max(lower_bound.abs());
    let sum_sq_sensitivity = max_val * max_val;
    let noisy_sum_sq = true_sum_sq + sample_laplace(sum_sq_sensitivity / eps_per_query);

    // Private count
    let true_count = values.len() as f64;
    let noisy_count = (true_count + sample_laplace(1.0 / eps_per_query)).max(2.0);

    // Update privacy accountant
    accountant.update(epsilon, 0.0);

    // Var = E[X^2] - E[X]^2 = (sum_sq / n) - (sum / n)^2
    let mean = noisy_sum / noisy_count;
    let mean_sq = noisy_sum_sq / noisy_count;
    let variance = (mean_sq - mean * mean).max(0.0); // Variance can't be negative

    Ok(variance)
}

/// Computes a differentially private histogram over the data.
///
/// # Arguments
///
/// * `values` - The array of values.
/// * `bins` - The bin edges (must be sorted, length = num_bins + 1).
/// * `epsilon` - The privacy budget for this query.
/// * `accountant` - The privacy accountant to track privacy loss.
///
/// # Returns
///
/// A vector of noisy counts for each bin.
///
/// # Example
///
/// ```rust
/// use differential_privacy::aggregations::private_histogram;
/// use differential_privacy::privacy_accounting::PrivacyAccountant;
/// use ndarray::array;
///
/// let values = array![1.0, 2.5, 3.0, 5.5, 7.0, 8.5, 9.0];
/// let bins = vec![0.0, 3.0, 6.0, 10.0]; // 3 bins: [0,3), [3,6), [6,10)
/// let mut accountant = PrivacyAccountant::new();
///
/// let noisy_counts = private_histogram(values.view(), &bins, 0.5, &mut accountant)
///     .expect("Invalid parameters");
/// println!("Histogram: {:?}", noisy_counts);
/// ```
pub fn private_histogram(
    values: ArrayView1<f64>,
    bins: &[f64],
    epsilon: f64,
    accountant: &mut PrivacyAccountant,
) -> Result<Vec<f64>, &'static str> {
    if epsilon <= 0.0 {
        return Err("Epsilon must be positive.");
    }
    if bins.len() < 2 {
        return Err("Need at least 2 bin edges.");
    }

    // Check bins are sorted
    for i in 1..bins.len() {
        if bins[i] < bins[i - 1] {
            return Err("Bin edges must be sorted.");
        }
    }

    let num_bins = bins.len() - 1;
    let mut counts = vec![0.0; num_bins];

    // Count values in each bin
    for &v in values.iter() {
        for i in 0..num_bins {
            if v >= bins[i] && v < bins[i + 1] {
                counts[i] += 1.0;
                break;
            }
        }
        // Handle right edge
        if v == bins[num_bins] {
            counts[num_bins - 1] += 1.0;
        }
    }

    // Add Laplace noise to each bin count
    // Sensitivity = 1 (one person affects one bin by at most 1)
    let sensitivity = 1.0;
    for count in &mut counts {
        *count = (*count + sample_laplace(sensitivity / epsilon)).max(0.0);
    }

    // Update privacy accountant
    accountant.update(epsilon, 0.0);

    Ok(counts)
}

/// Adds Laplace noise to each element of an array.
///
/// # Arguments
///
/// * `values` - The array of values.
/// * `sensitivity` - The L1 sensitivity of the query.
/// * `epsilon` - The privacy budget.
/// * `accountant` - The privacy accountant to track privacy loss.
///
/// # Returns
///
/// A new array with noise added to each element.
pub fn add_laplace_noise_vector(
    values: ArrayView1<f64>,
    sensitivity: f64,
    epsilon: f64,
    accountant: &mut PrivacyAccountant,
) -> Result<Array1<f64>, &'static str> {
    if epsilon <= 0.0 {
        return Err("Epsilon must be positive.");
    }
    if sensitivity < 0.0 {
        return Err("Sensitivity must be non-negative.");
    }

    let scale = sensitivity / epsilon;
    let noisy: Array1<f64> = values.mapv(|v| v + sample_laplace(scale));

    accountant.update(epsilon, 0.0);

    Ok(noisy)
}

/// Adds Gaussian noise to each element of an array.
///
/// # Arguments
///
/// * `values` - The array of values.
/// * `sensitivity` - The L2 sensitivity of the query.
/// * `epsilon` - The privacy budget.
/// * `delta` - The delta parameter for (ε,δ)-DP.
/// * `accountant` - The privacy accountant to track privacy loss.
///
/// # Returns
///
/// A new array with noise added to each element.
pub fn add_gaussian_noise_vector(
    values: ArrayView1<f64>,
    sensitivity: f64,
    epsilon: f64,
    delta: f64,
    accountant: &mut PrivacyAccountant,
) -> Result<Array1<f64>, &'static str> {
    if epsilon <= 0.0 {
        return Err("Epsilon must be positive.");
    }
    if sensitivity < 0.0 {
        return Err("Sensitivity must be non-negative.");
    }
    if delta <= 0.0 || delta >= 1.0 {
        return Err("Delta must be in (0, 1).");
    }

    let sigma = sensitivity * (2.0 * (1.25_f64 / delta).ln()).sqrt() / epsilon;
    let noisy: Array1<f64> = values.mapv(|v| v + sample_gaussian(sigma));

    accountant.update(epsilon, delta);

    Ok(noisy)
}

// ============================================================================
// Sensitivity Calculators
// ============================================================================

/// Calculates the sensitivity of a count query.
///
/// A count query has sensitivity 1: adding or removing one record changes the count by 1.
pub fn sensitivity_count() -> f64 {
    1.0
}

/// Calculates the sensitivity of a sum query over bounded values.
///
/// # Arguments
///
/// * `lower_bound` - The minimum possible value.
/// * `upper_bound` - The maximum possible value.
///
/// # Returns
///
/// The sensitivity of the sum query.
pub fn sensitivity_sum(lower_bound: f64, upper_bound: f64) -> f64 {
    upper_bound - lower_bound
}

/// Calculates the sensitivity of a mean query.
///
/// Note: The sensitivity of the mean depends on the dataset size, which may leak information.
/// For truly private mean, use `private_mean` which handles this properly.
///
/// # Arguments
///
/// * `lower_bound` - The minimum possible value.
/// * `upper_bound` - The maximum possible value.
/// * `n` - The number of records (assumed public).
///
/// # Returns
///
/// The sensitivity of the mean query.
pub fn sensitivity_mean(lower_bound: f64, upper_bound: f64, n: usize) -> f64 {
    if n == 0 {
        return f64::INFINITY;
    }
    (upper_bound - lower_bound) / n as f64
}

/// Calculates the L2 sensitivity for a vector query.
///
/// # Arguments
///
/// * `max_contribution` - The maximum L2 norm of any individual's contribution.
///
/// # Returns
///
/// The L2 sensitivity.
pub fn sensitivity_l2(max_contribution: f64) -> f64 {
    max_contribution
}

// ============================================================================
// Helper Functions
// ============================================================================

fn sample_laplace(scale: f64) -> f64 {
    use rand::Rng;
    if scale == 0.0 {
        return 0.0;
    }
    let uniform: f64 = rand::thread_rng().gen::<f64>() - 0.5;
    -(scale) * uniform.signum() * (1.0 - 2.0 * uniform.abs()).ln()
}

fn sample_gaussian(sigma: f64) -> f64 {
    use rand_distr::{Distribution, Normal};
    if sigma == 0.0 {
        return 0.0;
    }
    let normal = Normal::new(0.0, sigma).expect("Invalid sigma");
    normal.sample(&mut rand::thread_rng())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_private_sum() {
        let values = array![10.0, 20.0, 30.0, 40.0];
        let mut accountant = PrivacyAccountant::new();

        let result = private_sum(values.view(), 0.0, 100.0, 1.0, &mut accountant);
        assert!(result.is_ok());

        // True sum is 100, noisy sum should be in a reasonable range
        let noisy_sum = result.unwrap();
        assert!(noisy_sum.is_finite());

        // Check accountant was updated
        let (eps, _) = accountant.compute_basic_composition();
        assert!((eps - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_private_sum_with_clipping() {
        // Values outside bounds should be clipped
        let values = array![-50.0, 150.0, 50.0];
        let mut accountant = PrivacyAccountant::new();

        // With high epsilon (low noise), we can check clipping works
        let result = private_sum(values.view(), 0.0, 100.0, 100.0, &mut accountant);
        assert!(result.is_ok());

        // Clipped values: 0, 100, 50 -> sum = 150
        let noisy_sum = result.unwrap();
        assert!((noisy_sum - 150.0).abs() < 10.0); // Should be close to 150
    }

    #[test]
    fn test_private_count() {
        let values = array![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut accountant = PrivacyAccountant::new();

        // Count values > 25 (true count = 3)
        let result = private_count(values.view(), |&x| x > 25.0, 1.0, &mut accountant);
        assert!(result.is_ok());

        let noisy_count = result.unwrap();
        assert!(noisy_count >= 0.0); // Count can't be negative
        assert!(noisy_count.is_finite());
    }

    #[test]
    fn test_private_mean() {
        let values = array![10.0, 20.0, 30.0, 40.0];
        let mut accountant = PrivacyAccountant::new();

        // True mean = 25
        let result = private_mean(values.view(), 0.0, 100.0, 1.0, &mut accountant);
        assert!(result.is_ok());

        let noisy_mean = result.unwrap();
        assert!(noisy_mean.is_finite());
    }

    #[test]
    fn test_private_variance() {
        let values = array![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut accountant = PrivacyAccountant::new();

        let result = private_variance(values.view(), 0.0, 100.0, 1.0, &mut accountant);
        assert!(result.is_ok());

        let noisy_var = result.unwrap();
        assert!(noisy_var >= 0.0); // Variance can't be negative
    }

    #[test]
    fn test_private_histogram() {
        let values = array![1.0, 2.5, 3.0, 5.5, 7.0, 8.5, 9.0];
        let bins = vec![0.0, 3.0, 6.0, 10.0];
        let mut accountant = PrivacyAccountant::new();

        let result = private_histogram(values.view(), &bins, 1.0, &mut accountant);
        assert!(result.is_ok());

        let counts = result.unwrap();
        assert_eq!(counts.len(), 3);

        // All counts should be non-negative
        for count in &counts {
            assert!(*count >= 0.0);
        }
    }

    #[test]
    fn test_add_laplace_noise_vector() {
        let values = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut accountant = PrivacyAccountant::new();

        let result = add_laplace_noise_vector(values.view(), 1.0, 1.0, &mut accountant);
        assert!(result.is_ok());

        let noisy = result.unwrap();
        assert_eq!(noisy.len(), values.len());

        // Check all values are finite
        for v in noisy.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_add_gaussian_noise_vector() {
        let values = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut accountant = PrivacyAccountant::new();

        let result = add_gaussian_noise_vector(values.view(), 1.0, 1.0, 1e-5, &mut accountant);
        assert!(result.is_ok());

        let noisy = result.unwrap();
        assert_eq!(noisy.len(), values.len());

        // Check accountant tracks delta
        let (eps, delta) = accountant.compute_basic_composition();
        assert!((eps - 1.0).abs() < 1e-10);
        assert!((delta - 1e-5).abs() < 1e-15);
    }

    #[test]
    fn test_sensitivity_calculators() {
        assert_eq!(sensitivity_count(), 1.0);
        assert_eq!(sensitivity_sum(0.0, 100.0), 100.0);
        assert_eq!(sensitivity_mean(0.0, 100.0, 10), 10.0);
        assert_eq!(sensitivity_l2(5.0), 5.0);
    }

    #[test]
    fn test_private_sum_invalid_params() {
        let values = array![1.0, 2.0];
        let mut accountant = PrivacyAccountant::new();

        // Invalid epsilon
        let result = private_sum(values.view(), 0.0, 100.0, 0.0, &mut accountant);
        assert!(result.is_err());

        // Invalid bounds
        let result = private_sum(values.view(), 100.0, 0.0, 1.0, &mut accountant);
        assert!(result.is_err());
    }

    #[test]
    fn test_private_count_all() {
        let values = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut accountant = PrivacyAccountant::new();

        // Count all values (predicate always true)
        let result = private_count(values.view(), |_| true, 10.0, &mut accountant);
        assert!(result.is_ok());

        // With high epsilon, should be close to 5
        let noisy_count = result.unwrap();
        assert!((noisy_count - 5.0).abs() < 2.0);
    }

    #[test]
    fn test_private_histogram_edge_cases() {
        let values = array![0.0, 3.0, 6.0, 10.0]; // Values exactly on edges
        let bins = vec![0.0, 3.0, 6.0, 10.0];
        let mut accountant = PrivacyAccountant::new();

        let result = private_histogram(values.view(), &bins, 10.0, &mut accountant);
        assert!(result.is_ok());
    }

    #[test]
    fn test_empty_histogram_bins() {
        let values = array![1.0, 2.0];
        let bins = vec![0.0]; // Only one edge, invalid
        let mut accountant = PrivacyAccountant::new();

        let result = private_histogram(values.view(), &bins, 1.0, &mut accountant);
        assert!(result.is_err());
    }
}
