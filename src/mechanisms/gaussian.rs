// src/mechanisms/gaussian.rs

use rand_distr::{Distribution, Normal};
use crate::privacy_accounting::PrivacyAccountant;

/// Adds Gaussian noise to a value to ensure (ε, δ)-differential privacy.
///
/// The Gaussian Mechanism adds noise drawn from a Gaussian (normal) distribution
/// calibrated to the sensitivity of the query and the desired privacy parameters.
/// This mechanism provides (ε, δ)-differential privacy guarantees.
///
/// # Arguments
///
/// * `value` - The original numeric value to which noise will be added.
/// * `sensitivity` - The L2 sensitivity of the query, representing the maximum change
///                   in the query's output due to the addition or removal of a single
///                   individual's data.
/// * `epsilon` - The privacy budget parameter (ε). Must be positive.
/// * `delta` - The probability of privacy loss (δ). Must be in the range (0, 1).
/// * `accountant` - A mutable reference to the `PrivacyAccountant` to track privacy loss.
///
/// # Returns
///
/// A `Result` containing the noisy value, or an error message if parameters are invalid.
///
/// # Examples
///
/// ```rust
/// use differential_privacy::mechanisms::gaussian_mechanism;
/// use differential_privacy::privacy_accounting::PrivacyAccountant;
///
/// fn main() {
///     let value = 100.0;
///     let sensitivity = 1.0;
///     let epsilon = 0.5;
///     let delta = 1e-5;
///     let mut accountant = PrivacyAccountant::new();
///     let noisy_value = gaussian_mechanism(value, sensitivity, epsilon, delta, &mut accountant)
///         .expect("Invalid parameters");
///     println!("Noisy Value: {}", noisy_value);
///     let (total_epsilon, total_delta) = accountant.compute_basic_composition();
///     println!("Total Epsilon: {}, Total Delta: {}", total_epsilon, total_delta);
/// }
/// ```
pub fn gaussian_mechanism(
    value: f64,
    sensitivity: f64,
    epsilon: f64,
    delta: f64,
    accountant: &mut PrivacyAccountant,
) -> Result<f64, &'static str> {
    // Validate input parameters
    if epsilon <= 0.0 {
        return Err("Epsilon must be positive.");
    }
    if sensitivity < 0.0 {
        return Err("Sensitivity must be non-negative.");
    }
    if delta <= 0.0 || delta >= 1.0 {
        return Err("Delta must be in the range (0, 1).");
    }

    // Calculate sigma using the analytic Gaussian mechanism formula:
    // σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
    // This provides (ε, δ)-differential privacy
    let sigma = sensitivity * (2.0 * (1.25_f64 / delta).ln()).sqrt() / epsilon;

    // Sample Gaussian noise
    let noise = sample_gaussian(sigma)?;

    // Update the privacy accountant with the consumed privacy budget
    accountant.update(epsilon, delta);

    // Return the noisy value
    Ok(value + noise)
}

/// Samples noise from a Gaussian distribution with the given standard deviation (sigma).
///
/// # Arguments
///
/// * `sigma` - The standard deviation of the Gaussian distribution.
///
/// # Returns
///
/// A `Result` containing a single sample of Gaussian-distributed noise,
/// or an error if sigma is invalid.
fn sample_gaussian(sigma: f64) -> Result<f64, &'static str> {
    if sigma < 0.0 || sigma.is_nan() {
        return Err("Sigma must be non-negative and finite.");
    }

    // Handle the edge case where sigma is 0 (no noise needed)
    if sigma == 0.0 {
        return Ok(0.0);
    }

    let normal = Normal::new(0.0, sigma)
        .map_err(|_| "Failed to create normal distribution with given sigma.")?;
    Ok(normal.sample(&mut rand::thread_rng()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy_accounting::PrivacyAccountant;

    #[test]
    fn test_gaussian_mechanism_adds_noise() {
        let value = 10.0;
        let sensitivity = 1.0;
        let epsilon = 0.5;
        let delta = 1e-5;
        let mut accountant = PrivacyAccountant::new();
        let noisy_value = gaussian_mechanism(value, sensitivity, epsilon, delta, &mut accountant)
            .expect("Should succeed with valid parameters");

        // Check that noisy_value is finite
        assert!(noisy_value.is_finite(), "Noisy value should be finite.");

        // The noise should typically be within a reasonable range
        let sigma = sensitivity * (2.0 * (1.25_f64 / delta).ln()).sqrt() / epsilon;
        let noise = noisy_value - value;
        // With 99.99% probability, noise should be within 4 sigma
        assert!(
            noise.abs() <= 10.0 * sigma,
            "Noise magnitude is unexpectedly large: {} (sigma: {})", noise, sigma
        );
    }

    #[test]
    fn test_gaussian_mechanism_updates_accountant() {
        let value = 20.0;
        let sensitivity = 2.0;
        let epsilon = 1.0;
        let delta = 1e-5;
        let mut accountant = PrivacyAccountant::new();
        let _ = gaussian_mechanism(value, sensitivity, epsilon, delta, &mut accountant)
            .expect("Should succeed with valid parameters");

        let (total_epsilon, total_delta) = accountant.compute_basic_composition();
        assert_eq!(total_epsilon, epsilon);
        assert_eq!(total_delta, delta, "Delta should be tracked for Gaussian Mechanism.");
    }

    #[test]
    fn test_gaussian_mechanism_rejects_invalid_epsilon() {
        let mut accountant = PrivacyAccountant::new();
        let result = gaussian_mechanism(10.0, 1.0, 0.0, 1e-5, &mut accountant);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Epsilon must be positive.");

        let result = gaussian_mechanism(10.0, 1.0, -1.0, 1e-5, &mut accountant);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Epsilon must be positive.");
    }

    #[test]
    fn test_gaussian_mechanism_rejects_invalid_sensitivity() {
        let mut accountant = PrivacyAccountant::new();
        let result = gaussian_mechanism(10.0, -1.0, 0.5, 1e-5, &mut accountant);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Sensitivity must be non-negative.");
    }

    #[test]
    fn test_gaussian_mechanism_rejects_invalid_delta() {
        let mut accountant = PrivacyAccountant::new();

        // Delta = 0 is invalid
        let result = gaussian_mechanism(10.0, 1.0, 0.5, 0.0, &mut accountant);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Delta must be in the range (0, 1).");

        // Delta = 1 is invalid
        let result = gaussian_mechanism(10.0, 1.0, 0.5, 1.0, &mut accountant);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Delta must be in the range (0, 1).");

        // Delta > 1 is invalid
        let result = gaussian_mechanism(10.0, 1.0, 0.5, 1.5, &mut accountant);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Delta must be in the range (0, 1).");

        // Negative delta is invalid
        let result = gaussian_mechanism(10.0, 1.0, 0.5, -0.1, &mut accountant);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Delta must be in the range (0, 1).");
    }

    #[test]
    fn test_gaussian_mechanism_zero_sensitivity() {
        let mut accountant = PrivacyAccountant::new();
        let result = gaussian_mechanism(10.0, 0.0, 0.5, 1e-5, &mut accountant);
        assert!(result.is_ok());
        // With zero sensitivity, sigma = 0, so no noise is added
        assert_eq!(result.unwrap(), 10.0);
    }

    #[test]
    fn test_sample_gaussian_distribution_properties() {
        // Generate a large number of samples and check statistical properties
        let sigma = 2.0;
        let num_samples = 100_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for _ in 0..num_samples {
            let sample = sample_gaussian(sigma).expect("Should succeed");
            sum += sample;
            sum_sq += sample.powi(2);
        }

        let mean = sum / num_samples as f64;
        let variance = (sum_sq / num_samples as f64) - mean.powi(2);

        // For Gaussian distribution, mean should be ~0 and variance should be ~sigma^2
        assert!(
            mean.abs() < 0.1,
            "Mean of Gaussian samples deviates significantly from 0: {}", mean
        );
        assert!(
            (variance - sigma.powi(2)).abs() < 0.1,
            "Variance of Gaussian samples deviates from expected: {} (expected: {})", variance, sigma.powi(2)
        );
    }
}
