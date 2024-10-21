use rand_distr::{Distribution, Normal};

/// Adds Gaussian noise to a value based on differential privacy parameters.
///
/// # Arguments
///
/// * `value` - The original numeric value.
/// * `sensitivity` - The sensitivity of the query.
/// * `epsilon` - The privacy budget.
/// * `delta` - The allowable probability of privacy loss.
///
/// # Returns
///
/// The noisy value.
pub fn gaussian_mechanism(value: f64, sensitivity: f64, epsilon: f64, delta: f64) -> f64 {
    let sigma = (2.0 * sensitivity.powi(2) * (epsilon).ln().abs().recip()).sqrt(); // Simplified calculation
    value + sample_gaussian(sigma)
}

/// Samples noise from a Gaussian distribution with the given sigma.
fn sample_gaussian(sigma: f64) -> f64 {
    let normal = Normal::new(0.0, sigma).unwrap();
    normal.sample(&mut rand::thread_rng())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_mechanism() {
        let value = 10.0;
        let sensitivity = 1.0;
        let epsilon = 0.5;
        let delta = 1e-5;
        let noisy_value = gaussian_mechanism(value, sensitivity, epsilon, delta);
        assert!(noisy_value.is_finite());
    }
}