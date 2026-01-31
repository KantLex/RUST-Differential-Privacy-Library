// src/mechanisms/subsampling.rs

//! Privacy Amplification by Subsampling.
//!
//! When a differentially private mechanism is applied to a random subsample of the data,
//! the privacy guarantee is amplified. This module provides utilities for computing
//! amplified privacy parameters and for applying subsampled mechanisms.
//!
//! ## Key Concepts
//!
//! - **Poisson Subsampling**: Each record is included independently with probability q.
//! - **Uniform Subsampling**: A fixed-size subset is sampled uniformly at random.
//!
//! ## Privacy Amplification
//!
//! For a base mechanism with privacy parameter ε, subsampling with probability q
//! yields an amplified privacy guarantee. The amplification is significant when q is small.
//!
//! - For small ε: amplified ε ≈ q * ε
//! - General bound: amplified ε = ln(1 + q(e^ε - 1))
//!
//! ## Use Cases
//!
//! - **DP-SGD**: Privacy amplification from mini-batch sampling in machine learning.
//! - **Large-scale Analytics**: Stronger guarantees when analyzing data samples.
//! - **Streaming**: Processing random subsets of data streams.

use rand::Rng;
use ndarray::{Array1, ArrayView1};

use crate::privacy_accounting::PrivacyAccountant;
use crate::mechanisms::laplace::laplace_mechanism;

/// Computes the amplified epsilon for Poisson subsampling.
///
/// When each record is included independently with probability q,
/// and the base mechanism is ε-DP, the effective privacy guarantee
/// is amplified to ε' where ε' < ε.
///
/// # Arguments
///
/// * `base_epsilon` - The privacy parameter of the base mechanism.
/// * `sampling_probability` - The probability q that each record is included (0 < q ≤ 1).
///
/// # Returns
///
/// The amplified epsilon value, or an error if parameters are invalid.
///
/// # Formula
///
/// ε' = ln(1 + q(e^ε - 1))
///
/// For small ε, this approximates to ε' ≈ q * ε.
///
/// # Example
///
/// ```rust
/// use differential_privacy::mechanisms::amplify_epsilon_poisson;
///
/// // Base mechanism with ε=1.0, sampling 10% of data
/// let amplified = amplify_epsilon_poisson(1.0, 0.1).expect("Valid parameters");
/// println!("Amplified epsilon: {:.4}", amplified);  // Much smaller than 1.0
/// assert!(amplified < 1.0);
/// ```
pub fn amplify_epsilon_poisson(
    base_epsilon: f64,
    sampling_probability: f64,
) -> Result<f64, &'static str> {
    // Validate parameters
    if base_epsilon < 0.0 {
        return Err("Base epsilon must be non-negative.");
    }
    if sampling_probability <= 0.0 || sampling_probability > 1.0 {
        return Err("Sampling probability must be in (0, 1].");
    }

    // Special case: q = 1 means no subsampling
    if (sampling_probability - 1.0).abs() < 1e-10 {
        return Ok(base_epsilon);
    }

    // Special case: ε = 0 means perfect privacy
    if base_epsilon == 0.0 {
        return Ok(0.0);
    }

    // Amplification formula: ε' = ln(1 + q(e^ε - 1))
    let amplified = (1.0 + sampling_probability * (base_epsilon.exp() - 1.0)).ln();

    Ok(amplified)
}

/// Computes the amplified epsilon for uniform subsampling (without replacement).
///
/// When sampling k records uniformly at random from n total records,
/// the privacy guarantee is amplified.
///
/// # Arguments
///
/// * `base_epsilon` - The privacy parameter of the base mechanism.
/// * `sample_size` - The number of records to sample (k).
/// * `population_size` - The total number of records (n).
///
/// # Returns
///
/// The amplified epsilon value, or an error if parameters are invalid.
///
/// # Example
///
/// ```rust
/// use differential_privacy::mechanisms::amplify_epsilon_uniform;
///
/// // Base mechanism with ε=1.0, sampling 100 from 10000 records
/// let amplified = amplify_epsilon_uniform(1.0, 100, 10000).expect("Valid parameters");
/// println!("Amplified epsilon: {:.4}", amplified);
/// assert!(amplified < 1.0);
/// ```
pub fn amplify_epsilon_uniform(
    base_epsilon: f64,
    sample_size: usize,
    population_size: usize,
) -> Result<f64, &'static str> {
    if base_epsilon < 0.0 {
        return Err("Base epsilon must be non-negative.");
    }
    if sample_size == 0 {
        return Err("Sample size must be positive.");
    }
    if population_size == 0 {
        return Err("Population size must be positive.");
    }
    if sample_size > population_size {
        return Err("Sample size cannot exceed population size.");
    }

    // Sampling probability
    let q = sample_size as f64 / population_size as f64;

    // Use the Poisson amplification formula (tight for uniform sampling too)
    amplify_epsilon_poisson(base_epsilon, q)
}

/// Computes the amplified (ε, δ) for Poisson subsampling with approximate DP.
///
/// For (ε, δ)-DP mechanisms, subsampling amplifies both parameters.
///
/// # Arguments
///
/// * `base_epsilon` - The privacy parameter ε of the base mechanism.
/// * `base_delta` - The privacy parameter δ of the base mechanism.
/// * `sampling_probability` - The probability q that each record is included.
///
/// # Returns
///
/// A tuple (amplified_epsilon, amplified_delta), or an error if parameters are invalid.
///
/// # Example
///
/// ```rust
/// use differential_privacy::mechanisms::amplify_epsilon_delta_poisson;
///
/// let (amp_eps, amp_delta) = amplify_epsilon_delta_poisson(1.0, 1e-5, 0.01)
///     .expect("Valid parameters");
/// println!("Amplified: ε={:.4}, δ={:.2e}", amp_eps, amp_delta);
/// ```
pub fn amplify_epsilon_delta_poisson(
    base_epsilon: f64,
    base_delta: f64,
    sampling_probability: f64,
) -> Result<(f64, f64), &'static str> {
    if base_epsilon < 0.0 {
        return Err("Base epsilon must be non-negative.");
    }
    if base_delta < 0.0 || base_delta > 1.0 {
        return Err("Base delta must be in [0, 1].");
    }
    if sampling_probability <= 0.0 || sampling_probability > 1.0 {
        return Err("Sampling probability must be in (0, 1].");
    }

    // Amplify epsilon
    let amplified_epsilon = amplify_epsilon_poisson(base_epsilon, sampling_probability)?;

    // Amplify delta: δ' = q * δ
    let amplified_delta = sampling_probability * base_delta;

    Ok((amplified_epsilon, amplified_delta))
}

/// Computes the base epsilon needed to achieve a target amplified epsilon.
///
/// Given a target privacy level and sampling probability, computes what
/// base epsilon the mechanism should use.
///
/// # Arguments
///
/// * `target_epsilon` - The desired amplified privacy parameter.
/// * `sampling_probability` - The probability q that each record is included.
///
/// # Returns
///
/// The base epsilon needed, or an error if parameters are invalid.
///
/// # Example
///
/// ```rust
/// use differential_privacy::mechanisms::compute_base_epsilon;
///
/// // Want ε=0.1 after amplification with q=0.01
/// let base = compute_base_epsilon(0.1, 0.01).expect("Valid parameters");
/// println!("Need base epsilon: {:.2}", base);  // Much larger than 0.1
/// ```
pub fn compute_base_epsilon(
    target_epsilon: f64,
    sampling_probability: f64,
) -> Result<f64, &'static str> {
    if target_epsilon < 0.0 {
        return Err("Target epsilon must be non-negative.");
    }
    if sampling_probability <= 0.0 || sampling_probability > 1.0 {
        return Err("Sampling probability must be in (0, 1].");
    }

    // Special case: q = 1 means no amplification
    if (sampling_probability - 1.0).abs() < 1e-10 {
        return Ok(target_epsilon);
    }

    // Special case: target = 0
    if target_epsilon == 0.0 {
        return Ok(0.0);
    }

    // Inverse of amplification: base_ε = ln(1 + (e^ε' - 1) / q)
    let exp_target = target_epsilon.exp();
    let base = (1.0 + (exp_target - 1.0) / sampling_probability).ln();

    Ok(base)
}

/// Performs Poisson subsampling on a dataset.
///
/// Each element is included independently with probability q.
///
/// # Arguments
///
/// * `data` - The full dataset as an array view.
/// * `sampling_probability` - The probability q that each element is included.
///
/// # Returns
///
/// A vector of indices that were selected.
///
/// # Example
///
/// ```rust
/// use differential_privacy::mechanisms::poisson_subsample_indices;
/// use ndarray::array;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let indices = poisson_subsample_indices(data.view(), 0.3);
/// println!("Selected indices: {:?}", indices);
/// ```
pub fn poisson_subsample_indices(
    data: ArrayView1<f64>,
    sampling_probability: f64,
) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    let mut indices = Vec::new();

    for i in 0..data.len() {
        if rng.gen::<f64>() < sampling_probability {
            indices.push(i);
        }
    }

    indices
}

/// Performs Poisson subsampling on a dataset and returns the sampled values.
///
/// # Arguments
///
/// * `data` - The full dataset as an array view.
/// * `sampling_probability` - The probability q that each element is included.
///
/// # Returns
///
/// An array containing the sampled values.
pub fn poisson_subsample(
    data: ArrayView1<f64>,
    sampling_probability: f64,
) -> Array1<f64> {
    let indices = poisson_subsample_indices(data, sampling_probability);
    Array1::from_iter(indices.iter().map(|&i| data[i]))
}

/// Performs uniform subsampling (without replacement) on a dataset.
///
/// Selects exactly k elements uniformly at random.
///
/// # Arguments
///
/// * `data` - The full dataset as an array view.
/// * `sample_size` - The number of elements to sample.
///
/// # Returns
///
/// A tuple of (sampled_values, selected_indices), or an error if sample_size > data.len().
///
/// # Example
///
/// ```rust
/// use differential_privacy::mechanisms::uniform_subsample;
/// use ndarray::array;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let (sample, indices) = uniform_subsample(data.view(), 3).expect("Valid size");
/// println!("Sampled values: {:?}", sample);
/// println!("Indices: {:?}", indices);
/// ```
pub fn uniform_subsample(
    data: ArrayView1<f64>,
    sample_size: usize,
) -> Result<(Array1<f64>, Vec<usize>), &'static str> {
    if sample_size > data.len() {
        return Err("Sample size cannot exceed data size.");
    }
    if sample_size == 0 {
        return Ok((Array1::zeros(0), vec![]));
    }

    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..data.len()).collect();

    // Fisher-Yates shuffle (partial, up to sample_size)
    for i in 0..sample_size {
        let j = rng.gen_range(i..data.len());
        indices.swap(i, j);
    }

    indices.truncate(sample_size);
    indices.sort(); // Keep indices in order for consistency

    let sampled = Array1::from_iter(indices.iter().map(|&i| data[i]));

    Ok((sampled, indices))
}

/// Configuration for a subsampled mechanism.
#[derive(Debug, Clone)]
pub struct SubsampledMechanism {
    /// The sampling probability (for Poisson) or sample_size/population_size (for uniform).
    pub sampling_probability: f64,
    /// The base epsilon of the underlying mechanism.
    pub base_epsilon: f64,
    /// The base delta of the underlying mechanism (0 for pure ε-DP).
    pub base_delta: f64,
    /// The amplified epsilon after accounting for subsampling.
    pub amplified_epsilon: f64,
    /// The amplified delta after accounting for subsampling.
    pub amplified_delta: f64,
}

impl SubsampledMechanism {
    /// Creates a new subsampled mechanism configuration with Poisson subsampling.
    ///
    /// # Arguments
    ///
    /// * `base_epsilon` - The epsilon of the base mechanism.
    /// * `base_delta` - The delta of the base mechanism (0 for pure ε-DP).
    /// * `sampling_probability` - The probability each record is included.
    ///
    /// # Example
    ///
    /// ```rust
    /// use differential_privacy::mechanisms::SubsampledMechanism;
    ///
    /// let mech = SubsampledMechanism::new_poisson(1.0, 0.0, 0.01)
    ///     .expect("Valid parameters");
    /// println!("Base ε: {}, Amplified ε: {:.4}", mech.base_epsilon, mech.amplified_epsilon);
    /// ```
    pub fn new_poisson(
        base_epsilon: f64,
        base_delta: f64,
        sampling_probability: f64,
    ) -> Result<Self, &'static str> {
        let (amplified_epsilon, amplified_delta) =
            amplify_epsilon_delta_poisson(base_epsilon, base_delta, sampling_probability)?;

        Ok(Self {
            sampling_probability,
            base_epsilon,
            base_delta,
            amplified_epsilon,
            amplified_delta,
        })
    }

    /// Creates a new subsampled mechanism configuration with uniform subsampling.
    ///
    /// # Arguments
    ///
    /// * `base_epsilon` - The epsilon of the base mechanism.
    /// * `base_delta` - The delta of the base mechanism.
    /// * `sample_size` - The number of records to sample.
    /// * `population_size` - The total number of records.
    pub fn new_uniform(
        base_epsilon: f64,
        base_delta: f64,
        sample_size: usize,
        population_size: usize,
    ) -> Result<Self, &'static str> {
        if sample_size > population_size {
            return Err("Sample size cannot exceed population size.");
        }
        if population_size == 0 {
            return Err("Population size must be positive.");
        }

        let sampling_probability = sample_size as f64 / population_size as f64;
        Self::new_poisson(base_epsilon, base_delta, sampling_probability)
    }

    /// Returns the amplification factor (base_epsilon / amplified_epsilon).
    ///
    /// A higher value indicates stronger amplification.
    pub fn amplification_factor(&self) -> f64 {
        if self.amplified_epsilon == 0.0 {
            f64::INFINITY
        } else {
            self.base_epsilon / self.amplified_epsilon
        }
    }
}

/// Applies the Laplace mechanism to a subsampled sum query.
///
/// This function:
/// 1. Subsamples the data with probability q
/// 2. Computes the sum of the subsample
/// 3. Adds Laplace noise calibrated to the base sensitivity
/// 4. Records the amplified privacy loss
///
/// # Arguments
///
/// * `data` - The full dataset.
/// * `sensitivity` - The sensitivity of the sum query (e.g., max value).
/// * `base_epsilon` - The epsilon for the Laplace mechanism.
/// * `sampling_probability` - The probability each record is included.
/// * `accountant` - Privacy accountant to track the amplified privacy loss.
///
/// # Returns
///
/// The noisy sum of the subsample.
///
/// # Example
///
/// ```rust
/// use differential_privacy::mechanisms::subsampled_laplace_sum;
/// use differential_privacy::privacy_accounting::PrivacyAccountant;
/// use ndarray::array;
///
/// let mut accountant = PrivacyAccountant::new();
/// let data = array![10.0, 20.0, 30.0, 40.0, 50.0];
///
/// let noisy_sum = subsampled_laplace_sum(
///     data.view(), 50.0, 1.0, 0.5, &mut accountant
/// ).expect("Valid parameters");
///
/// let (eps, _) = accountant.compute_basic_composition();
/// println!("Noisy sum: {:.2}, Privacy cost: {:.4}", noisy_sum, eps);
/// ```
pub fn subsampled_laplace_sum(
    data: ArrayView1<f64>,
    sensitivity: f64,
    base_epsilon: f64,
    sampling_probability: f64,
    accountant: &mut PrivacyAccountant,
) -> Result<f64, &'static str> {
    if sensitivity <= 0.0 {
        return Err("Sensitivity must be positive.");
    }
    if base_epsilon <= 0.0 {
        return Err("Base epsilon must be positive.");
    }
    if sampling_probability <= 0.0 || sampling_probability > 1.0 {
        return Err("Sampling probability must be in (0, 1].");
    }

    // Subsample the data
    let subsample = poisson_subsample(data, sampling_probability);

    // Compute the sum
    let sum: f64 = subsample.iter().sum();

    // Compute amplified epsilon
    let amplified_epsilon = amplify_epsilon_poisson(base_epsilon, sampling_probability)?;

    // Add Laplace noise using base epsilon (the amplification is accounted for)
    // We create a temporary accountant for the base mechanism
    let mut temp_accountant = PrivacyAccountant::new();
    let noisy_sum = laplace_mechanism(sum, sensitivity, base_epsilon, &mut temp_accountant);

    // Record the amplified privacy loss
    accountant.update(amplified_epsilon, 0.0);

    Ok(noisy_sum)
}

/// Applies the Laplace mechanism to a subsampled mean query.
///
/// This function:
/// 1. Subsamples the data with probability q
/// 2. Computes the mean of the subsample (with noise on both sum and count)
/// 3. Records the amplified privacy loss
///
/// # Arguments
///
/// * `data` - The full dataset.
/// * `lower_bound` - Lower bound for data values (for sensitivity).
/// * `upper_bound` - Upper bound for data values.
/// * `base_epsilon` - The total epsilon for the mechanism.
/// * `sampling_probability` - The probability each record is included.
/// * `accountant` - Privacy accountant to track privacy loss.
///
/// # Returns
///
/// The noisy mean of the subsample.
pub fn subsampled_laplace_mean(
    data: ArrayView1<f64>,
    lower_bound: f64,
    upper_bound: f64,
    base_epsilon: f64,
    sampling_probability: f64,
    accountant: &mut PrivacyAccountant,
) -> Result<f64, &'static str> {
    if lower_bound >= upper_bound {
        return Err("Lower bound must be less than upper bound.");
    }
    if base_epsilon <= 0.0 {
        return Err("Base epsilon must be positive.");
    }
    if sampling_probability <= 0.0 || sampling_probability > 1.0 {
        return Err("Sampling probability must be in (0, 1].");
    }

    // Subsample the data
    let subsample = poisson_subsample(data, sampling_probability);

    if subsample.is_empty() {
        // No data sampled, return midpoint
        accountant.update(0.0, 0.0);
        return Ok((lower_bound + upper_bound) / 2.0);
    }

    // Clip values to bounds
    let clipped: Array1<f64> = subsample.mapv(|x| x.clamp(lower_bound, upper_bound));

    // Split epsilon between sum and count
    let eps_sum = base_epsilon / 2.0;
    let eps_count = base_epsilon / 2.0;

    // Compute noisy sum
    let sum: f64 = clipped.iter().sum();
    let sum_sensitivity = upper_bound - lower_bound;
    let mut temp_accountant = PrivacyAccountant::new();
    let noisy_sum = laplace_mechanism(sum, sum_sensitivity, eps_sum, &mut temp_accountant);

    // Compute noisy count
    let count = clipped.len() as f64;
    let noisy_count = laplace_mechanism(count, 1.0, eps_count, &mut temp_accountant);

    // Compute mean (protect against division by zero or negative count)
    let safe_count = noisy_count.max(1.0);
    let noisy_mean = noisy_sum / safe_count;

    // Clip result to valid range
    let result = noisy_mean.clamp(lower_bound, upper_bound);

    // Compute amplified epsilon and record
    let amplified_epsilon = amplify_epsilon_poisson(base_epsilon, sampling_probability)?;
    accountant.update(amplified_epsilon, 0.0);

    Ok(result)
}

/// Computes the expected noise reduction from subsampling.
///
/// For a given target privacy level, subsampling allows using a larger
/// base epsilon, which means less noise.
///
/// # Arguments
///
/// * `sampling_probability` - The probability each record is included.
///
/// # Returns
///
/// The approximate factor by which noise can be reduced (for small ε).
///
/// # Example
///
/// ```rust
/// use differential_privacy::mechanisms::subsampling_noise_reduction;
///
/// let reduction = subsampling_noise_reduction(0.01);
/// println!("Noise reduction factor: {:.0}x", reduction);  // ~100x
/// ```
pub fn subsampling_noise_reduction(sampling_probability: f64) -> f64 {
    // For small ε, amplification is approximately q * ε
    // So base ε can be approximately 1/q times larger
    // Which means noise scale (Δ/ε) is approximately q times smaller
    // But we're computing the sum of q*n elements, so noise relative to signal
    // is actually reduced by factor of approximately 1/sqrt(q) for averaging

    // For sum queries with fixed sensitivity, noise reduction is approximately 1/q
    1.0 / sampling_probability
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_amplify_epsilon_poisson_basic() {
        // Small q should give much smaller amplified epsilon
        let amplified = amplify_epsilon_poisson(1.0, 0.01).unwrap();
        assert!(amplified < 0.02, "Amplified epsilon should be much smaller");
        assert!(amplified > 0.0, "Amplified epsilon should be positive");
    }

    #[test]
    fn test_amplify_epsilon_poisson_q_equals_1() {
        // q = 1 means no subsampling, no amplification
        let amplified = amplify_epsilon_poisson(1.0, 1.0).unwrap();
        assert!((amplified - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_amplify_epsilon_poisson_zero_epsilon() {
        let amplified = amplify_epsilon_poisson(0.0, 0.5).unwrap();
        assert_eq!(amplified, 0.0);
    }

    #[test]
    fn test_amplify_epsilon_poisson_invalid_params() {
        assert!(amplify_epsilon_poisson(-1.0, 0.5).is_err());
        assert!(amplify_epsilon_poisson(1.0, 0.0).is_err());
        assert!(amplify_epsilon_poisson(1.0, 1.5).is_err());
        assert!(amplify_epsilon_poisson(1.0, -0.1).is_err());
    }

    #[test]
    fn test_amplify_epsilon_uniform() {
        let amplified = amplify_epsilon_uniform(1.0, 100, 10000).unwrap();
        assert!(amplified < 0.02);
    }

    #[test]
    fn test_amplify_epsilon_uniform_invalid_params() {
        assert!(amplify_epsilon_uniform(1.0, 0, 100).is_err());
        assert!(amplify_epsilon_uniform(1.0, 100, 0).is_err());
        assert!(amplify_epsilon_uniform(1.0, 200, 100).is_err());
    }

    #[test]
    fn test_amplify_epsilon_delta_poisson() {
        let (amp_eps, amp_delta) = amplify_epsilon_delta_poisson(1.0, 1e-5, 0.01).unwrap();
        assert!(amp_eps < 1.0);
        assert!((amp_delta - 1e-7).abs() < 1e-10);
    }

    #[test]
    fn test_compute_base_epsilon() {
        // Round-trip test
        let base = 2.0;
        let q = 0.1;
        let amplified = amplify_epsilon_poisson(base, q).unwrap();
        let recovered = compute_base_epsilon(amplified, q).unwrap();
        assert!((recovered - base).abs() < 1e-10);
    }

    #[test]
    fn test_compute_base_epsilon_no_amplification() {
        let base = compute_base_epsilon(1.0, 1.0).unwrap();
        assert!((base - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_poisson_subsample_indices() {
        let data = Array1::from_vec((0..1000).map(|x| x as f64).collect());

        // With q = 0.1, expect roughly 100 samples
        let indices = poisson_subsample_indices(data.view(), 0.1);
        assert!(indices.len() > 50 && indices.len() < 150,
            "Expected ~100 samples, got {}", indices.len());
    }

    #[test]
    fn test_poisson_subsample() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample = poisson_subsample(data.view(), 1.0);
        assert_eq!(sample.len(), 5, "q=1 should include all elements");
    }

    #[test]
    fn test_uniform_subsample() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (sample, indices) = uniform_subsample(data.view(), 3).unwrap();

        assert_eq!(sample.len(), 3);
        assert_eq!(indices.len(), 3);

        // Verify indices are valid and in order
        for i in 0..indices.len() - 1 {
            assert!(indices[i] < indices[i + 1]);
        }

        // Verify samples match data
        for (i, &idx) in indices.iter().enumerate() {
            assert_eq!(sample[i], data[idx]);
        }
    }

    #[test]
    fn test_uniform_subsample_invalid() {
        let data = array![1.0, 2.0, 3.0];
        assert!(uniform_subsample(data.view(), 5).is_err());
    }

    #[test]
    fn test_uniform_subsample_empty() {
        let data = array![1.0, 2.0, 3.0];
        let (sample, indices) = uniform_subsample(data.view(), 0).unwrap();
        assert!(sample.is_empty());
        assert!(indices.is_empty());
    }

    #[test]
    fn test_subsampled_mechanism_poisson() {
        let mech = SubsampledMechanism::new_poisson(1.0, 0.0, 0.01).unwrap();
        assert_eq!(mech.base_epsilon, 1.0);
        assert!(mech.amplified_epsilon < 0.02);
        assert!(mech.amplification_factor() > 50.0);
    }

    #[test]
    fn test_subsampled_mechanism_uniform() {
        let mech = SubsampledMechanism::new_uniform(1.0, 1e-5, 100, 10000).unwrap();
        assert!(mech.amplified_epsilon < 0.02);
        assert!(mech.amplified_delta < 1e-6);
    }

    #[test]
    fn test_subsampled_laplace_sum() {
        let mut accountant = PrivacyAccountant::new();
        let data = Array1::from_vec((0..100).map(|x| x as f64).collect());

        let result = subsampled_laplace_sum(data.view(), 100.0, 1.0, 0.1, &mut accountant);
        assert!(result.is_ok());

        // Check that amplified epsilon was recorded
        let (eps, _) = accountant.compute_basic_composition();
        assert!(eps < 1.0, "Should record amplified epsilon, not base");
    }

    #[test]
    fn test_subsampled_laplace_sum_invalid() {
        let mut accountant = PrivacyAccountant::new();
        let data = array![1.0, 2.0, 3.0];

        assert!(subsampled_laplace_sum(data.view(), 0.0, 1.0, 0.5, &mut accountant).is_err());
        assert!(subsampled_laplace_sum(data.view(), 1.0, 0.0, 0.5, &mut accountant).is_err());
        assert!(subsampled_laplace_sum(data.view(), 1.0, 1.0, 0.0, &mut accountant).is_err());
    }

    #[test]
    fn test_subsampled_laplace_mean() {
        let mut accountant = PrivacyAccountant::new();
        let data = Array1::from_vec((0..100).map(|x| x as f64).collect());

        let result = subsampled_laplace_mean(data.view(), 0.0, 100.0, 1.0, 0.5, &mut accountant);
        assert!(result.is_ok());

        let mean = result.unwrap();
        assert!(mean >= 0.0 && mean <= 100.0, "Mean should be in bounds");
    }

    #[test]
    fn test_subsampled_laplace_mean_empty_sample() {
        let mut accountant = PrivacyAccountant::new();
        let data = array![50.0];

        // With very low probability, might get empty sample
        // Function should handle this gracefully
        for _ in 0..10 {
            let result = subsampled_laplace_mean(data.view(), 0.0, 100.0, 1.0, 0.01, &mut accountant);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_subsampling_noise_reduction() {
        let reduction = subsampling_noise_reduction(0.01);
        assert!((reduction - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_amplification_increases_with_smaller_q() {
        let eps1 = amplify_epsilon_poisson(1.0, 0.1).unwrap();
        let eps2 = amplify_epsilon_poisson(1.0, 0.01).unwrap();
        let eps3 = amplify_epsilon_poisson(1.0, 0.001).unwrap();

        // Smaller q should give smaller amplified epsilon
        assert!(eps1 > eps2);
        assert!(eps2 > eps3);
    }

    #[test]
    fn test_amplification_linear_for_small_epsilon() {
        // For small ε, amplification should be approximately q * ε
        let base_eps = 0.01;
        let q = 0.1;
        let amplified = amplify_epsilon_poisson(base_eps, q).unwrap();

        // Should be close to q * base_eps for small values
        let expected = q * base_eps;
        let relative_error = (amplified - expected).abs() / expected;
        assert!(relative_error < 0.1, "Amplification should be ~linear for small ε");
    }
}
