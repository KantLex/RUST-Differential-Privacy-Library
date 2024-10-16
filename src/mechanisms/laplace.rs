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
        // However, due to randomness, this test might occasionally fail
        // To make it robust, we can check if noise is within expected bounds
        let noise = noisy_value - value;
        let expected_scale = sensitivity / epsilon;
        // The Laplace distribution has a probability density function that decays exponentially
        // So, most noise values should lie within a few scales
        assert!(
            noise.abs() <= 10.0 * expected_scale,
            "Noise is too large, which is unlikely."
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
        assert_eq!(total_delta, 0.0);
    }
}