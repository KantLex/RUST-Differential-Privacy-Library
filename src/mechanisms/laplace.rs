// src/mechanisms/laplace.rs

use rand::Rng;
use crate::privacy_accounting::PrivacyAccountant;

/// Adds Laplace noise to a numeric value to ensure differential privacy.
///
/// The Laplace Mechanism adds noise drawn from a Laplace distribution
/// calibrated to the sensitivity of the query and the desired privacy budget (epsilon).
///
/// # Arguments
///
/// * `value` - The original numeric value to which noise will be added.
/// * `sensitivity` - The sensitivity of the query, representing the maximum change
///                   in the query's output due to the addition or removal of a single
///                   individual's data.
/// * `epsilon` - The privacy budget parameter (ε), controlling the trade-off between
///               privacy and accuracy. Lower ε provides stronger privacy guarantees.
/// * `accountant` - A mutable reference to the `PrivacyAccountant` to track privacy loss.
///
/// # Returns
///
/// The noisy value after adding Laplace-distributed noise.
///
/// # Examples
///
/// ```rust
/// use differential_privacy::mechanisms::laplace_mechanism;
/// use differential_privacy::privacy_accounting::PrivacyAccountant;
///
/// fn main() {
///     let value = 100.0;
///     let sensitivity = 1.0;
///     let epsilon = 0.5;
///     let mut accountant = PrivacyAccountant::new(0.0, 0.0);
///     let noisy_value = laplace_mechanism(value, sensitivity, epsilon, &mut accountant);
///     println!("Noisy Value: {}", noisy_value);
///     let (total_epsilon, _) = accountant.get_privacy_loss();
///     println!("Total Epsilon: {}", total_epsilon);
/// }
/// ```
pub fn laplace_mechanism(
    value: f64,
    sensitivity: f64,
    epsilon: f64,
    accountant: &mut PrivacyAccountant,
) -> f64 {
    // Calculate the scale parameter for the Laplace distribution
    let scale = sensitivity / epsilon;

    // Sample Laplace noise
    let noise = sample_laplace(scale);

    // Update the privacy accountant with the consumed epsilon
    accountant.update(epsilon, 0.0);

    // Return the noisy value
    value + noise
}

/// Samples noise from a Laplace distribution with the specified scale.
///
/// # Arguments
///
/// * `scale` - The scale parameter (b) of the Laplace distribution.
///
/// # Returns
///
/// A single sample of Laplace-distributed noise.
fn sample_laplace(scale: f64) -> f64 {
    let uniform: f64 = rand::thread_rng().gen::<f64>() - 0.5;
    -(scale) * uniform.signum() * (1.0 - 2.0 * uniform.abs()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy_accounting::PrivacyAccountant;

    #[test]
    fn test_laplace_mechanism_adds_noise() {
        let value = 10.0;
        let sensitivity = 1.0;
        let epsilon = 0.5;
        let mut accountant = PrivacyAccountant::new(0.0, 0.0);
        let noisy_value = laplace_mechanism(value, sensitivity, epsilon, &mut accountant);

        // Check that noisy_value is not equal to the original value
        // (with high probability, since noise is random)
        assert!(noisy_value != value, "Noisy value should differ from original value.");

        // Check that noise is within expected range with high probability
        let noise = noisy_value - value;
        let scale = sensitivity / epsilon;
        // The probability that |noise| > k * scale decreases exponentially
        // For testing, we can check if |noise| <= 10 * scale
        assert!(
            noise.abs() <= 10.0 * scale,
            "Noise magnitude is too large, which is unlikely."
        );
    }

    #[test]
    fn test_laplace_mechanism_updates_accountant() {
        let value = 20.0;
        let sensitivity = 2.0;
        let epsilon = 1.0;
        let mut accountant = PrivacyAccountant::new(0.0, 0.0);
        let _ = laplace_mechanism(value, sensitivity, epsilon, &mut accountant);
        let (total_epsilon, total_delta) = accountant.get_privacy_loss();
        assert_eq!(total_epsilon, epsilon);
        assert_eq!(total_delta, 0.0, "Delta should remain zero for Laplace Mechanism.");
    }

    #[test]
    fn test_sample_laplace_distribution_properties() {
        // Generate a large number of samples and check statistical properties
        let scale = 1.0;
        let num_samples = 100_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for _ in 0..num_samples {
            let sample = sample_laplace(scale);
            sum += sample;
            sum_sq += sample.powi(2);
        }

        let mean = sum / num_samples as f64;
        let variance = (sum_sq / num_samples as f64) - mean.powi(2);

        // For Laplace distribution, mean should be ~0 and variance should be ~2 * scale^2
        assert!(
            mean.abs() < 0.1,
            "Mean of Laplace samples deviates significantly from 0."
        );
        assert!(
            (variance - 2.0 * scale.powi(2)).abs() < 0.1,
            "Variance of Laplace samples deviates significantly from expected value."
        );
    }
}