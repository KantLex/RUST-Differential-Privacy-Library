// src/mechanisms/exponential.rs

use rand::Rng;
use crate::privacy_accounting::PrivacyAccountant;

/// Selects an item from a set of candidates using the Exponential Mechanism.
///
/// The Exponential Mechanism is used for privately selecting an item from a discrete
/// set of candidates. Each candidate is assigned a utility score, and the mechanism
/// selects items with probability proportional to exp(ε × utility / (2 × sensitivity)).
///
/// # Arguments
///
/// * `utilities` - A slice of utility scores for each candidate. Higher scores indicate
///                 more desirable candidates.
/// * `sensitivity` - The sensitivity of the utility function, representing the maximum
///                   change in utility due to adding or removing a single individual's data.
/// * `epsilon` - The privacy budget parameter (ε). Must be positive.
/// * `accountant` - A mutable reference to the `PrivacyAccountant` to track privacy loss.
///
/// # Returns
///
/// A `Result` containing the index of the selected candidate, or an error message
/// if parameters are invalid.
///
/// # Examples
///
/// ```rust
/// use differential_privacy::mechanisms::exponential_mechanism;
/// use differential_privacy::privacy_accounting::PrivacyAccountant;
///
/// fn main() {
///     // Suppose we want to select the best category privately
///     let utilities = vec![10.0, 25.0, 15.0, 5.0]; // Category scores
///     let sensitivity = 1.0;
///     let epsilon = 0.5;
///     let mut accountant = PrivacyAccountant::new();
///
///     let selected_index = exponential_mechanism(&utilities, sensitivity, epsilon, &mut accountant)
///         .expect("Invalid parameters");
///     println!("Selected category index: {}", selected_index);
/// }
/// ```
pub fn exponential_mechanism(
    utilities: &[f64],
    sensitivity: f64,
    epsilon: f64,
    accountant: &mut PrivacyAccountant,
) -> Result<usize, &'static str> {
    // Validate input parameters
    if utilities.is_empty() {
        return Err("Utilities slice must not be empty.");
    }
    if epsilon <= 0.0 {
        return Err("Epsilon must be positive.");
    }
    if sensitivity <= 0.0 {
        return Err("Sensitivity must be positive.");
    }

    // Check for NaN or infinite values in utilities
    for &u in utilities {
        if u.is_nan() || u.is_infinite() {
            return Err("Utilities must be finite numbers.");
        }
    }

    // Calculate the selection probabilities using the exponential mechanism formula
    // P(i) ∝ exp(ε × u(i) / (2 × Δu))
    let scaling_factor = epsilon / (2.0 * sensitivity);

    // Find max utility for numerical stability (subtract before exponentiating)
    let max_utility = utilities
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    // Calculate unnormalized probabilities (weights)
    let weights: Vec<f64> = utilities
        .iter()
        .map(|&u| ((u - max_utility) * scaling_factor).exp())
        .collect();

    // Calculate the sum of weights for normalization
    let total_weight: f64 = weights.iter().sum();

    if total_weight <= 0.0 || total_weight.is_nan() || total_weight.is_infinite() {
        return Err("Failed to compute valid probability weights.");
    }

    // Sample from the distribution
    let selected_index = sample_from_weights(&weights, total_weight)?;

    // Update the privacy accountant
    accountant.update(epsilon, 0.0);

    Ok(selected_index)
}

/// Samples an index from a weighted distribution.
///
/// # Arguments
///
/// * `weights` - The unnormalized weights for each item.
/// * `total_weight` - The sum of all weights.
///
/// # Returns
///
/// The index of the sampled item.
fn sample_from_weights(weights: &[f64], total_weight: f64) -> Result<usize, &'static str> {
    let mut rng = rand::thread_rng();
    let sample: f64 = rng.gen::<f64>() * total_weight;

    let mut cumulative = 0.0;
    for (i, &weight) in weights.iter().enumerate() {
        cumulative += weight;
        if sample <= cumulative {
            return Ok(i);
        }
    }

    // Due to floating-point precision, we might not have returned yet
    // Return the last index as a fallback
    Ok(weights.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy_accounting::PrivacyAccountant;
    use std::collections::HashMap;

    #[test]
    fn test_exponential_mechanism_selects_valid_index() {
        let utilities = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sensitivity = 1.0;
        let epsilon = 1.0;
        let mut accountant = PrivacyAccountant::new();

        let result = exponential_mechanism(&utilities, sensitivity, epsilon, &mut accountant);
        assert!(result.is_ok());
        let index = result.unwrap();
        assert!(index < utilities.len(), "Selected index should be within bounds.");
    }

    #[test]
    fn test_exponential_mechanism_updates_accountant() {
        let utilities = vec![1.0, 2.0, 3.0];
        let sensitivity = 1.0;
        let epsilon = 0.5;
        let mut accountant = PrivacyAccountant::new();

        let _ = exponential_mechanism(&utilities, sensitivity, epsilon, &mut accountant)
            .expect("Should succeed");

        let (total_epsilon, total_delta) = accountant.compute_basic_composition();
        assert_eq!(total_epsilon, epsilon);
        assert_eq!(total_delta, 0.0, "Delta should be zero for Exponential Mechanism.");
    }

    #[test]
    fn test_exponential_mechanism_prefers_higher_utility() {
        // With high epsilon, the mechanism should strongly prefer higher utility items
        let utilities = vec![0.0, 0.0, 100.0, 0.0]; // Index 2 has much higher utility
        let sensitivity = 1.0;
        let epsilon = 10.0; // High epsilon = less noise = stronger preference
        let mut accountant = PrivacyAccountant::new();

        let mut counts = HashMap::new();
        let num_trials = 1000;

        for _ in 0..num_trials {
            let index = exponential_mechanism(&utilities, sensitivity, epsilon, &mut accountant)
                .expect("Should succeed");
            *counts.entry(index).or_insert(0) += 1;
        }

        // With high epsilon and such disparity, index 2 should be selected most often
        let index_2_count = *counts.get(&2).unwrap_or(&0);
        assert!(
            index_2_count > num_trials / 2,
            "Index 2 should be selected more than half the time, but was selected {} times",
            index_2_count
        );
    }

    #[test]
    fn test_exponential_mechanism_rejects_empty_utilities() {
        let utilities: Vec<f64> = vec![];
        let mut accountant = PrivacyAccountant::new();

        let result = exponential_mechanism(&utilities, 1.0, 0.5, &mut accountant);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Utilities slice must not be empty.");
    }

    #[test]
    fn test_exponential_mechanism_rejects_invalid_epsilon() {
        let utilities = vec![1.0, 2.0];
        let mut accountant = PrivacyAccountant::new();

        let result = exponential_mechanism(&utilities, 1.0, 0.0, &mut accountant);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Epsilon must be positive.");

        let result = exponential_mechanism(&utilities, 1.0, -1.0, &mut accountant);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Epsilon must be positive.");
    }

    #[test]
    fn test_exponential_mechanism_rejects_invalid_sensitivity() {
        let utilities = vec![1.0, 2.0];
        let mut accountant = PrivacyAccountant::new();

        let result = exponential_mechanism(&utilities, 0.0, 0.5, &mut accountant);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Sensitivity must be positive.");

        let result = exponential_mechanism(&utilities, -1.0, 0.5, &mut accountant);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Sensitivity must be positive.");
    }

    #[test]
    fn test_exponential_mechanism_rejects_nan_utilities() {
        let utilities = vec![1.0, f64::NAN, 3.0];
        let mut accountant = PrivacyAccountant::new();

        let result = exponential_mechanism(&utilities, 1.0, 0.5, &mut accountant);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Utilities must be finite numbers.");
    }

    #[test]
    fn test_exponential_mechanism_handles_negative_utilities() {
        // Negative utilities are valid and should work correctly
        let utilities = vec![-10.0, -5.0, -1.0];
        let sensitivity = 1.0;
        let epsilon = 1.0;
        let mut accountant = PrivacyAccountant::new();

        let result = exponential_mechanism(&utilities, sensitivity, epsilon, &mut accountant);
        assert!(result.is_ok());
    }

    #[test]
    fn test_exponential_mechanism_single_candidate() {
        // With only one candidate, it should always be selected
        let utilities = vec![5.0];
        let sensitivity = 1.0;
        let epsilon = 0.5;
        let mut accountant = PrivacyAccountant::new();

        for _ in 0..10 {
            let index = exponential_mechanism(&utilities, sensitivity, epsilon, &mut accountant)
                .expect("Should succeed");
            assert_eq!(index, 0, "Single candidate should always be selected.");
        }
    }

    #[test]
    fn test_exponential_mechanism_equal_utilities() {
        // With equal utilities, selection should be approximately uniform
        let utilities = vec![5.0, 5.0, 5.0, 5.0];
        let sensitivity = 1.0;
        let epsilon = 1.0;
        let mut accountant = PrivacyAccountant::new();

        let mut counts = [0usize; 4];
        let num_trials = 10000;

        for _ in 0..num_trials {
            let index = exponential_mechanism(&utilities, sensitivity, epsilon, &mut accountant)
                .expect("Should succeed");
            counts[index] += 1;
        }

        // Each index should be selected approximately 25% of the time
        let expected = num_trials / 4;
        for (i, &count) in counts.iter().enumerate() {
            let deviation = (count as f64 - expected as f64).abs() / expected as f64;
            assert!(
                deviation < 0.1,
                "Index {} selected {} times, expected ~{} (deviation: {:.2}%)",
                i, count, expected, deviation * 100.0
            );
        }
    }
}
