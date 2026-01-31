// src/mechanisms/report_noisy_max.rs

use rand::Rng;
use crate::privacy_accounting::PrivacyAccountant;

/// Reports the index of the maximum value from a set of counts using differential privacy.
///
/// The Report Noisy Max mechanism adds Laplace noise to each count and returns the index
/// of the maximum noisy value. This provides ε-differential privacy for releasing the
/// argmax of a set of counts.
///
/// # Arguments
///
/// * `counts` - A slice of count values. Each count represents a frequency or score.
/// * `sensitivity` - The sensitivity of each count, representing the maximum change
///                   due to adding or removing a single individual's data.
///                   For counting queries, this is typically 1.0.
/// * `epsilon` - The privacy budget parameter (ε). Must be positive.
/// * `accountant` - A mutable reference to the `PrivacyAccountant` to track privacy loss.
///
/// # Returns
///
/// A `Result` containing a tuple of (index, noisy_max_value), where index is the
/// position of the maximum noisy count, or an error message if parameters are invalid.
///
/// # Examples
///
/// ```rust
/// use differential_privacy::mechanisms::report_noisy_max;
/// use differential_privacy::privacy_accounting::PrivacyAccountant;
///
/// fn main() {
///     // Suppose we have vote counts for different options
///     let counts = vec![150.0, 200.0, 175.0, 50.0];
///     let sensitivity = 1.0; // Each person affects one count by at most 1
///     let epsilon = 0.5;
///     let mut accountant = PrivacyAccountant::new();
///
///     let (winning_index, noisy_max) = report_noisy_max(&counts, sensitivity, epsilon, &mut accountant)
///         .expect("Invalid parameters");
///     println!("Winning option index: {} (noisy count: {})", winning_index, noisy_max);
/// }
/// ```
pub fn report_noisy_max(
    counts: &[f64],
    sensitivity: f64,
    epsilon: f64,
    accountant: &mut PrivacyAccountant,
) -> Result<(usize, f64), &'static str> {
    // Validate input parameters
    if counts.is_empty() {
        return Err("Counts slice must not be empty.");
    }
    if epsilon <= 0.0 {
        return Err("Epsilon must be positive.");
    }
    if sensitivity <= 0.0 {
        return Err("Sensitivity must be positive.");
    }

    // Check for NaN or infinite values in counts
    for &c in counts {
        if c.is_nan() || c.is_infinite() {
            return Err("Counts must be finite numbers.");
        }
    }

    // Calculate the scale for Laplace noise
    let scale = sensitivity / epsilon;

    // Add Laplace noise to each count and find the max
    let mut max_index = 0;
    let mut max_noisy_value = f64::NEG_INFINITY;

    for (i, &count) in counts.iter().enumerate() {
        let noise = sample_laplace(scale);
        let noisy_count = count + noise;

        if noisy_count > max_noisy_value {
            max_noisy_value = noisy_count;
            max_index = i;
        }
    }

    // Update the privacy accountant
    accountant.update(epsilon, 0.0);

    Ok((max_index, max_noisy_value))
}

/// Reports only the index of the maximum value (without the noisy value).
///
/// This is a convenience wrapper around `report_noisy_max` that only returns
/// the index, which is often sufficient for many use cases.
///
/// # Arguments
///
/// * `counts` - A slice of count values.
/// * `sensitivity` - The sensitivity of each count.
/// * `epsilon` - The privacy budget parameter (ε).
/// * `accountant` - A mutable reference to the `PrivacyAccountant`.
///
/// # Returns
///
/// A `Result` containing the index of the maximum noisy count.
pub fn report_noisy_argmax(
    counts: &[f64],
    sensitivity: f64,
    epsilon: f64,
    accountant: &mut PrivacyAccountant,
) -> Result<usize, &'static str> {
    let (index, _) = report_noisy_max(counts, sensitivity, epsilon, accountant)?;
    Ok(index)
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
    use std::collections::HashMap;

    #[test]
    fn test_report_noisy_max_returns_valid_index() {
        let counts = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let sensitivity = 1.0;
        let epsilon = 1.0;
        let mut accountant = PrivacyAccountant::new();

        let result = report_noisy_max(&counts, sensitivity, epsilon, &mut accountant);
        assert!(result.is_ok());
        let (index, noisy_max) = result.unwrap();
        assert!(index < counts.len(), "Selected index should be within bounds.");
        assert!(noisy_max.is_finite(), "Noisy max should be finite.");
    }

    #[test]
    fn test_report_noisy_max_updates_accountant() {
        let counts = vec![10.0, 20.0, 30.0];
        let sensitivity = 1.0;
        let epsilon = 0.5;
        let mut accountant = PrivacyAccountant::new();

        let _ = report_noisy_max(&counts, sensitivity, epsilon, &mut accountant)
            .expect("Should succeed");

        let (total_epsilon, total_delta) = accountant.compute_basic_composition();
        assert_eq!(total_epsilon, epsilon);
        assert_eq!(total_delta, 0.0, "Delta should be zero for Report Noisy Max.");
    }

    #[test]
    fn test_report_noisy_max_prefers_higher_counts() {
        // With high epsilon and significant differences, the true max should win often
        let counts = vec![10.0, 10.0, 1000.0, 10.0]; // Index 2 has much higher count
        let sensitivity = 1.0;
        let epsilon = 10.0; // High epsilon = less noise

        let mut counts_map = HashMap::new();
        let num_trials = 1000;

        for _ in 0..num_trials {
            let mut accountant = PrivacyAccountant::new();
            let (index, _) = report_noisy_max(&counts, sensitivity, epsilon, &mut accountant)
                .expect("Should succeed");
            *counts_map.entry(index).or_insert(0) += 1;
        }

        // With such disparity and high epsilon, index 2 should win almost always
        let index_2_count = *counts_map.get(&2).unwrap_or(&0);
        assert!(
            index_2_count > num_trials * 9 / 10,
            "Index 2 should be selected more than 90% of the time, but was selected {} times",
            index_2_count
        );
    }

    #[test]
    fn test_report_noisy_max_rejects_empty_counts() {
        let counts: Vec<f64> = vec![];
        let mut accountant = PrivacyAccountant::new();

        let result = report_noisy_max(&counts, 1.0, 0.5, &mut accountant);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Counts slice must not be empty.");
    }

    #[test]
    fn test_report_noisy_max_rejects_invalid_epsilon() {
        let counts = vec![10.0, 20.0];
        let mut accountant = PrivacyAccountant::new();

        let result = report_noisy_max(&counts, 1.0, 0.0, &mut accountant);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Epsilon must be positive.");

        let result = report_noisy_max(&counts, 1.0, -1.0, &mut accountant);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Epsilon must be positive.");
    }

    #[test]
    fn test_report_noisy_max_rejects_invalid_sensitivity() {
        let counts = vec![10.0, 20.0];
        let mut accountant = PrivacyAccountant::new();

        let result = report_noisy_max(&counts, 0.0, 0.5, &mut accountant);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Sensitivity must be positive.");

        let result = report_noisy_max(&counts, -1.0, 0.5, &mut accountant);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Sensitivity must be positive.");
    }

    #[test]
    fn test_report_noisy_max_rejects_nan_counts() {
        let counts = vec![10.0, f64::NAN, 30.0];
        let mut accountant = PrivacyAccountant::new();

        let result = report_noisy_max(&counts, 1.0, 0.5, &mut accountant);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Counts must be finite numbers.");
    }

    #[test]
    fn test_report_noisy_argmax() {
        let counts = vec![100.0, 200.0, 150.0];
        let sensitivity = 1.0;
        let epsilon = 5.0;
        let mut accountant = PrivacyAccountant::new();

        let result = report_noisy_argmax(&counts, sensitivity, epsilon, &mut accountant);
        assert!(result.is_ok());
        let index = result.unwrap();
        assert!(index < counts.len());
    }

    #[test]
    fn test_report_noisy_max_single_count() {
        // With only one count, it should always be selected
        let counts = vec![42.0];
        let sensitivity = 1.0;
        let epsilon = 0.5;
        let mut accountant = PrivacyAccountant::new();

        for _ in 0..10 {
            let (index, _) = report_noisy_max(&counts, sensitivity, epsilon, &mut accountant)
                .expect("Should succeed");
            assert_eq!(index, 0, "Single count should always be selected.");
        }
    }

    #[test]
    fn test_report_noisy_max_equal_counts_distribution() {
        // With equal counts and low epsilon, selection should show randomness
        let counts = vec![50.0, 50.0, 50.0, 50.0];
        let sensitivity = 1.0;
        let epsilon = 0.1; // Low epsilon = more noise

        let mut counts_map = [0usize; 4];
        let num_trials = 10000;

        for _ in 0..num_trials {
            let mut accountant = PrivacyAccountant::new();
            let (index, _) = report_noisy_max(&counts, sensitivity, epsilon, &mut accountant)
                .expect("Should succeed");
            counts_map[index] += 1;
        }

        // Each index should be selected approximately 25% of the time
        let expected = num_trials / 4;
        for (i, &count) in counts_map.iter().enumerate() {
            let deviation = (count as f64 - expected as f64).abs() / expected as f64;
            assert!(
                deviation < 0.15,
                "Index {} selected {} times, expected ~{} (deviation: {:.2}%)",
                i, count, expected, deviation * 100.0
            );
        }
    }

    #[test]
    fn test_report_noisy_max_handles_negative_counts() {
        // Negative counts are valid and should work correctly
        let counts = vec![-100.0, -50.0, -10.0];
        let sensitivity = 1.0;
        let epsilon = 1.0;
        let mut accountant = PrivacyAccountant::new();

        let result = report_noisy_max(&counts, sensitivity, epsilon, &mut accountant);
        assert!(result.is_ok());
    }
}
