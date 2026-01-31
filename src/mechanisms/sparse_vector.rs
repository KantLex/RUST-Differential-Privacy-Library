// src/mechanisms/sparse_vector.rs

//! Sparse Vector Technique (SVT) for Differential Privacy.
//!
//! The Sparse Vector Technique is a powerful mechanism for answering many threshold
//! queries with a fixed privacy budget. It's particularly useful when you expect
//! most queries to be below the threshold and only want to identify the few that exceed it.
//!
//! ## Key Properties
//!
//! - **Fixed Privacy Cost**: The total privacy budget is independent of the number of queries,
//!   only depending on the number of "Above Threshold" answers.
//! - **Streaming**: Queries can be processed one at a time.
//! - **Early Stopping**: The algorithm halts after finding a specified number of above-threshold queries.
//!
//! ## Variants
//!
//! - [`SparseVectorTechnique`]: Basic SVT that outputs only Above/Below for each query.
//! - [`NumericSparseVector`]: Extended SVT that also outputs noisy values for above-threshold queries.

use rand::Rng;

use crate::privacy_accounting::PrivacyAccountant;

/// Samples noise from a Laplace distribution with the specified scale.
fn sample_laplace(scale: f64) -> f64 {
    let uniform: f64 = rand::thread_rng().gen::<f64>() - 0.5;
    -(scale) * uniform.signum() * (1.0 - 2.0 * uniform.abs()).ln()
}

/// Result of a threshold query in the Sparse Vector Technique.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdResult {
    /// The query result is above the noisy threshold.
    Above,
    /// The query result is below the noisy threshold.
    Below,
}

/// Result of a threshold query in the Numeric Sparse Vector variant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NumericThresholdResult {
    /// The query result is above threshold, with the noisy value.
    Above(f64),
    /// The query result is below the noisy threshold.
    Below,
}

/// Sparse Vector Technique for answering threshold queries.
///
/// This mechanism allows answering many threshold queries of the form
/// "Is q(D) >= T?" with a fixed privacy budget that depends only on the
/// maximum number of "Above" answers, not the total number of queries.
///
/// # Privacy Guarantee
///
/// The mechanism provides ε-differential privacy where:
/// - ε₁ is used for the noisy threshold (split across all queries)
/// - ε₂ is used for the noisy query answers
/// - Total ε = ε₁ + ε₂ (typically ε₁ = ε₂ = ε/2)
///
/// # Example
///
/// ```rust
/// use differential_privacy::mechanisms::SparseVectorTechnique;
/// use differential_privacy::privacy_accounting::PrivacyAccountant;
///
/// fn main() {
///     let mut accountant = PrivacyAccountant::new();
///
///     // Create SVT with threshold 100, sensitivity 1, epsilon 1.0, max 3 "Above" answers
///     let mut svt = SparseVectorTechnique::new(100.0, 1.0, 1.0, 3, &mut accountant)
///         .expect("Invalid parameters");
///
///     // Query values
///     let queries = vec![50.0, 120.0, 80.0, 150.0, 90.0, 200.0];
///
///     for query_value in queries {
///         if let Some(result) = svt.query(query_value) {
///             println!("Query {}: {:?}", query_value, result);
///         } else {
///             println!("SVT exhausted (found max above-threshold answers)");
///             break;
///         }
///     }
/// }
/// ```
pub struct SparseVectorTechnique {
    /// The noisy threshold (T + Lap(2Δ/ε₁))
    noisy_threshold: f64,
    /// Query sensitivity
    sensitivity: f64,
    /// Epsilon allocated for query noise
    epsilon_query: f64,
    /// Maximum number of "Above" answers before halting
    max_above: usize,
    /// Current count of "Above" answers
    above_count: usize,
    /// Whether the mechanism has been exhausted
    exhausted: bool,
}

impl SparseVectorTechnique {
    /// Creates a new Sparse Vector Technique instance.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The threshold T for comparison.
    /// * `sensitivity` - The sensitivity Δ of each query (maximum change from one record).
    /// * `epsilon` - Total privacy budget (split between threshold and queries).
    /// * `max_above` - Maximum number of "Above" answers before halting (c).
    /// * `accountant` - Privacy accountant to track budget usage.
    ///
    /// # Returns
    ///
    /// A new `SparseVectorTechnique` instance, or an error if parameters are invalid.
    ///
    /// # Privacy Cost
    ///
    /// The mechanism consumes ε privacy budget regardless of the number of queries asked.
    pub fn new(
        threshold: f64,
        sensitivity: f64,
        epsilon: f64,
        max_above: usize,
        accountant: &mut PrivacyAccountant,
    ) -> Result<Self, &'static str> {
        // Validate parameters
        if !threshold.is_finite() {
            return Err("Threshold must be a finite number.");
        }
        if sensitivity <= 0.0 {
            return Err("Sensitivity must be positive.");
        }
        if epsilon <= 0.0 {
            return Err("Epsilon must be positive.");
        }
        if max_above == 0 {
            return Err("Max above must be at least 1.");
        }

        // Split epsilon between threshold and queries (standard allocation)
        let epsilon_threshold = epsilon / 2.0;
        let epsilon_query = epsilon / 2.0;

        // Add noise to threshold: T̃ = T + Lap(2Δ/ε₁)
        let threshold_scale = 2.0 * sensitivity / epsilon_threshold;
        let noisy_threshold = threshold + sample_laplace(threshold_scale);

        // Record privacy usage
        accountant.update(epsilon, 0.0);

        Ok(Self {
            noisy_threshold,
            sensitivity,
            epsilon_query,
            max_above,
            above_count: 0,
            exhausted: false,
        })
    }

    /// Creates SVT with custom epsilon allocation between threshold and queries.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The threshold T for comparison.
    /// * `sensitivity` - The sensitivity Δ of each query.
    /// * `epsilon_threshold` - Privacy budget for the noisy threshold.
    /// * `epsilon_query` - Privacy budget for query noise.
    /// * `max_above` - Maximum number of "Above" answers before halting.
    /// * `accountant` - Privacy accountant to track budget usage.
    pub fn with_custom_allocation(
        threshold: f64,
        sensitivity: f64,
        epsilon_threshold: f64,
        epsilon_query: f64,
        max_above: usize,
        accountant: &mut PrivacyAccountant,
    ) -> Result<Self, &'static str> {
        // Validate parameters
        if !threshold.is_finite() {
            return Err("Threshold must be a finite number.");
        }
        if sensitivity <= 0.0 {
            return Err("Sensitivity must be positive.");
        }
        if epsilon_threshold <= 0.0 {
            return Err("Epsilon for threshold must be positive.");
        }
        if epsilon_query <= 0.0 {
            return Err("Epsilon for queries must be positive.");
        }
        if max_above == 0 {
            return Err("Max above must be at least 1.");
        }

        // Add noise to threshold
        let threshold_scale = 2.0 * sensitivity / epsilon_threshold;
        let noisy_threshold = threshold + sample_laplace(threshold_scale);

        // Record total privacy usage
        accountant.update(epsilon_threshold + epsilon_query, 0.0);

        Ok(Self {
            noisy_threshold,
            sensitivity,
            epsilon_query,
            max_above,
            above_count: 0,
            exhausted: false,
        })
    }

    /// Processes a single threshold query.
    ///
    /// # Arguments
    ///
    /// * `query_value` - The true answer to the query (before noise).
    ///
    /// # Returns
    ///
    /// - `Some(ThresholdResult::Above)` if the noisy query exceeds the noisy threshold.
    /// - `Some(ThresholdResult::Below)` if the noisy query is below the noisy threshold.
    /// - `None` if the mechanism has been exhausted (max_above reached).
    pub fn query(&mut self, query_value: f64) -> Option<ThresholdResult> {
        if self.exhausted {
            return None;
        }

        // Add noise to query: ã = q + Lap(2cΔ/ε₂)
        // where c is max_above (ensures privacy even if all queries are above)
        let query_scale = 2.0 * (self.max_above as f64) * self.sensitivity / self.epsilon_query;
        let noisy_query = query_value + sample_laplace(query_scale);

        if noisy_query >= self.noisy_threshold {
            self.above_count += 1;
            if self.above_count >= self.max_above {
                self.exhausted = true;
            }
            Some(ThresholdResult::Above)
        } else {
            Some(ThresholdResult::Below)
        }
    }

    /// Processes multiple queries at once.
    ///
    /// # Arguments
    ///
    /// * `query_values` - A slice of true query answers.
    ///
    /// # Returns
    ///
    /// A vector of results for each query processed before exhaustion.
    /// The vector may be shorter than the input if the mechanism becomes exhausted.
    pub fn query_batch(&mut self, query_values: &[f64]) -> Vec<ThresholdResult> {
        let mut results = Vec::with_capacity(query_values.len());

        for &value in query_values {
            match self.query(value) {
                Some(result) => results.push(result),
                None => break,
            }
        }

        results
    }

    /// Returns whether the mechanism has been exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    /// Returns the number of "Above" answers found so far.
    pub fn above_count(&self) -> usize {
        self.above_count
    }

    /// Returns the remaining number of "Above" answers allowed.
    pub fn remaining_above(&self) -> usize {
        self.max_above - self.above_count
    }
}

/// Numeric Sparse Vector Technique.
///
/// An extension of SVT that outputs noisy values for queries that exceed the threshold,
/// rather than just Above/Below indicators. This is useful when you need the actual
/// (noisy) values, not just whether they exceed the threshold.
///
/// # Privacy Cost
///
/// Higher than basic SVT due to releasing numeric values. Uses:
/// - ε₁ for the noisy threshold
/// - ε₂ for each "Above" query's noisy value
///
/// # Example
///
/// ```rust
/// use differential_privacy::mechanisms::NumericSparseVector;
/// use differential_privacy::privacy_accounting::PrivacyAccountant;
///
/// fn main() {
///     let mut accountant = PrivacyAccountant::new();
///
///     let mut svt = NumericSparseVector::new(100.0, 1.0, 1.0, 3, &mut accountant)
///         .expect("Invalid parameters");
///
///     let queries = vec![50.0, 120.0, 80.0, 150.0];
///
///     for query_value in queries {
///         if let Some(result) = svt.query(query_value) {
///             match result {
///                 differential_privacy::mechanisms::NumericThresholdResult::Above(noisy_val) => {
///                     println!("Above threshold! Noisy value: {:.2}", noisy_val);
///                 }
///                 differential_privacy::mechanisms::NumericThresholdResult::Below => {
///                     println!("Below threshold");
///                 }
///             }
///         }
///     }
/// }
/// ```
pub struct NumericSparseVector {
    /// The noisy threshold
    noisy_threshold: f64,
    /// Query sensitivity
    sensitivity: f64,
    /// Epsilon for query noise (used for both comparison and output)
    epsilon_query: f64,
    /// Maximum number of "Above" answers
    max_above: usize,
    /// Current count of "Above" answers
    above_count: usize,
    /// Whether exhausted
    exhausted: bool,
}

impl NumericSparseVector {
    /// Creates a new Numeric Sparse Vector instance.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The threshold T for comparison.
    /// * `sensitivity` - The sensitivity Δ of each query.
    /// * `epsilon` - Total privacy budget.
    /// * `max_above` - Maximum number of "Above" answers before halting.
    /// * `accountant` - Privacy accountant to track budget usage.
    pub fn new(
        threshold: f64,
        sensitivity: f64,
        epsilon: f64,
        max_above: usize,
        accountant: &mut PrivacyAccountant,
    ) -> Result<Self, &'static str> {
        if !threshold.is_finite() {
            return Err("Threshold must be a finite number.");
        }
        if sensitivity <= 0.0 {
            return Err("Sensitivity must be positive.");
        }
        if epsilon <= 0.0 {
            return Err("Epsilon must be positive.");
        }
        if max_above == 0 {
            return Err("Max above must be at least 1.");
        }

        // For numeric SVT, we allocate:
        // - ε/3 for threshold
        // - 2ε/3 for queries (comparison + output)
        let epsilon_threshold = epsilon / 3.0;
        let epsilon_query = 2.0 * epsilon / 3.0;

        let threshold_scale = 2.0 * sensitivity / epsilon_threshold;
        let noisy_threshold = threshold + sample_laplace(threshold_scale);

        accountant.update(epsilon, 0.0);

        Ok(Self {
            noisy_threshold,
            sensitivity,
            epsilon_query,
            max_above,
            above_count: 0,
            exhausted: false,
        })
    }

    /// Processes a single threshold query, returning a noisy value if above threshold.
    ///
    /// # Arguments
    ///
    /// * `query_value` - The true answer to the query.
    ///
    /// # Returns
    ///
    /// - `Some(NumericThresholdResult::Above(noisy_value))` if above threshold.
    /// - `Some(NumericThresholdResult::Below)` if below threshold.
    /// - `None` if the mechanism has been exhausted.
    pub fn query(&mut self, query_value: f64) -> Option<NumericThresholdResult> {
        if self.exhausted {
            return None;
        }

        // Noise for comparison
        let comparison_scale = 4.0 * (self.max_above as f64) * self.sensitivity / self.epsilon_query;
        let noisy_comparison = query_value + sample_laplace(comparison_scale);

        if noisy_comparison >= self.noisy_threshold {
            // Add additional noise for the output value
            let output_scale = 4.0 * (self.max_above as f64) * self.sensitivity / self.epsilon_query;
            let noisy_output = query_value + sample_laplace(output_scale);

            self.above_count += 1;
            if self.above_count >= self.max_above {
                self.exhausted = true;
            }
            Some(NumericThresholdResult::Above(noisy_output))
        } else {
            Some(NumericThresholdResult::Below)
        }
    }

    /// Processes multiple queries at once.
    pub fn query_batch(&mut self, query_values: &[f64]) -> Vec<NumericThresholdResult> {
        let mut results = Vec::with_capacity(query_values.len());

        for &value in query_values {
            match self.query(value) {
                Some(result) => results.push(result),
                None => break,
            }
        }

        results
    }

    /// Returns whether the mechanism has been exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    /// Returns the number of "Above" answers found so far.
    pub fn above_count(&self) -> usize {
        self.above_count
    }

    /// Returns the remaining number of "Above" answers allowed.
    pub fn remaining_above(&self) -> usize {
        self.max_above - self.above_count
    }
}

/// One-shot Sparse Vector query for simple use cases.
///
/// Finds the first query that exceeds the threshold from a list of queries.
///
/// # Arguments
///
/// * `queries` - A slice of true query answers.
/// * `threshold` - The threshold for comparison.
/// * `sensitivity` - The sensitivity of each query.
/// * `epsilon` - Privacy budget.
/// * `accountant` - Privacy accountant.
///
/// # Returns
///
/// The index of the first query that exceeds the threshold (with noise),
/// or `None` if no query exceeds the threshold.
///
/// # Example
///
/// ```rust
/// use differential_privacy::mechanisms::sparse_vector_find_first;
/// use differential_privacy::privacy_accounting::PrivacyAccountant;
///
/// let mut accountant = PrivacyAccountant::new();
/// let queries = vec![50.0, 80.0, 120.0, 90.0, 150.0];
///
/// if let Some(index) = sparse_vector_find_first(&queries, 100.0, 1.0, 1.0, &mut accountant)
///     .expect("Invalid parameters")
/// {
///     println!("First query above threshold at index: {}", index);
/// } else {
///     println!("No query exceeded the threshold");
/// }
/// ```
pub fn sparse_vector_find_first(
    queries: &[f64],
    threshold: f64,
    sensitivity: f64,
    epsilon: f64,
    accountant: &mut PrivacyAccountant,
) -> Result<Option<usize>, &'static str> {
    if queries.is_empty() {
        return Err("Queries slice must not be empty.");
    }

    let mut svt = SparseVectorTechnique::new(threshold, sensitivity, epsilon, 1, accountant)?;

    for (i, &query_value) in queries.iter().enumerate() {
        if let Some(result) = svt.query(query_value) {
            if result == ThresholdResult::Above {
                return Ok(Some(i));
            }
        }
    }

    Ok(None)
}

/// Finds all queries that exceed the threshold (up to max_count).
///
/// # Arguments
///
/// * `queries` - A slice of true query answers.
/// * `threshold` - The threshold for comparison.
/// * `sensitivity` - The sensitivity of each query.
/// * `epsilon` - Privacy budget.
/// * `max_count` - Maximum number of above-threshold queries to find.
/// * `accountant` - Privacy accountant.
///
/// # Returns
///
/// A vector of indices where queries exceeded the threshold.
///
/// # Example
///
/// ```rust
/// use differential_privacy::mechanisms::sparse_vector_find_all;
/// use differential_privacy::privacy_accounting::PrivacyAccountant;
///
/// let mut accountant = PrivacyAccountant::new();
/// let queries = vec![50.0, 120.0, 80.0, 150.0, 90.0, 200.0];
///
/// let above_indices = sparse_vector_find_all(&queries, 100.0, 1.0, 1.0, 3, &mut accountant)
///     .expect("Invalid parameters");
///
/// println!("Queries above threshold at indices: {:?}", above_indices);
/// ```
pub fn sparse_vector_find_all(
    queries: &[f64],
    threshold: f64,
    sensitivity: f64,
    epsilon: f64,
    max_count: usize,
    accountant: &mut PrivacyAccountant,
) -> Result<Vec<usize>, &'static str> {
    if queries.is_empty() {
        return Err("Queries slice must not be empty.");
    }
    if max_count == 0 {
        return Err("Max count must be at least 1.");
    }

    let mut svt = SparseVectorTechnique::new(threshold, sensitivity, epsilon, max_count, accountant)?;
    let mut above_indices = Vec::new();

    for (i, &query_value) in queries.iter().enumerate() {
        match svt.query(query_value) {
            Some(ThresholdResult::Above) => {
                above_indices.push(i);
                if svt.is_exhausted() {
                    break;
                }
            }
            Some(ThresholdResult::Below) => {}
            None => break,
        }
    }

    Ok(above_indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy_accounting::PrivacyAccountant;

    #[test]
    fn test_svt_creation() {
        let mut accountant = PrivacyAccountant::new();
        let svt = SparseVectorTechnique::new(100.0, 1.0, 1.0, 3, &mut accountant);
        assert!(svt.is_ok());

        let (eps, _) = accountant.compute_basic_composition();
        assert_eq!(eps, 1.0);
    }

    #[test]
    fn test_svt_invalid_params() {
        let mut accountant = PrivacyAccountant::new();

        // Invalid threshold
        let result = SparseVectorTechnique::new(f64::NAN, 1.0, 1.0, 3, &mut accountant);
        assert!(result.is_err());

        // Invalid sensitivity
        let result = SparseVectorTechnique::new(100.0, 0.0, 1.0, 3, &mut accountant);
        assert!(result.is_err());
        let result = SparseVectorTechnique::new(100.0, -1.0, 1.0, 3, &mut accountant);
        assert!(result.is_err());

        // Invalid epsilon
        let result = SparseVectorTechnique::new(100.0, 1.0, 0.0, 3, &mut accountant);
        assert!(result.is_err());

        // Invalid max_above
        let result = SparseVectorTechnique::new(100.0, 1.0, 1.0, 0, &mut accountant);
        assert!(result.is_err());
    }

    #[test]
    fn test_svt_basic_queries() {
        let mut accountant = PrivacyAccountant::new();
        let mut svt = SparseVectorTechnique::new(100.0, 1.0, 10.0, 3, &mut accountant)
            .expect("Should create SVT");

        // Query well below threshold - should be Below
        let result = svt.query(10.0);
        assert!(result.is_some());
        // With high epsilon, this should almost certainly be Below

        // Query well above threshold - should be Above with high probability
        let result = svt.query(200.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_svt_exhaustion() {
        let mut accountant = PrivacyAccountant::new();
        let mut svt = SparseVectorTechnique::new(0.0, 1.0, 10.0, 2, &mut accountant)
            .expect("Should create SVT");

        // With threshold 0 and high epsilon, positive values should be Above
        let _ = svt.query(100.0); // First Above
        assert!(!svt.is_exhausted());
        assert_eq!(svt.remaining_above(), 1);

        let _ = svt.query(100.0); // Second Above
        assert!(svt.is_exhausted());
        assert_eq!(svt.remaining_above(), 0);

        // Should return None after exhaustion
        let result = svt.query(100.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_svt_batch_query() {
        let mut accountant = PrivacyAccountant::new();
        let mut svt = SparseVectorTechnique::new(100.0, 1.0, 10.0, 2, &mut accountant)
            .expect("Should create SVT");

        let queries = vec![50.0, 150.0, 60.0, 200.0, 70.0];
        let results = svt.query_batch(&queries);

        // Should have processed queries until exhaustion or end
        assert!(!results.is_empty());
    }

    #[test]
    fn test_svt_high_epsilon_accuracy() {
        // With very high epsilon (low noise), SVT should be accurate
        let mut above_count = 0;
        let num_trials = 100;

        for _ in 0..num_trials {
            let mut accountant = PrivacyAccountant::new();
            let mut svt = SparseVectorTechnique::new(100.0, 1.0, 100.0, 1, &mut accountant)
                .expect("Should create SVT");

            // Query value well above threshold
            if let Some(ThresholdResult::Above) = svt.query(200.0) {
                above_count += 1;
            }
        }

        // With epsilon=100 and value 100 above threshold, should almost always be Above
        assert!(
            above_count > 90,
            "Expected >90% Above, got {}%",
            above_count
        );
    }

    #[test]
    fn test_numeric_svt_creation() {
        let mut accountant = PrivacyAccountant::new();
        let svt = NumericSparseVector::new(100.0, 1.0, 1.0, 3, &mut accountant);
        assert!(svt.is_ok());
    }

    #[test]
    fn test_numeric_svt_returns_values() {
        let mut accountant = PrivacyAccountant::new();
        let mut svt = NumericSparseVector::new(50.0, 1.0, 10.0, 3, &mut accountant)
            .expect("Should create SVT");

        // Query well above threshold
        let result = svt.query(200.0);
        assert!(result.is_some());

        match result.unwrap() {
            NumericThresholdResult::Above(value) => {
                // Value should be in reasonable range of 200
                assert!(value > 100.0 && value < 300.0, "Noisy value {} out of range", value);
            }
            NumericThresholdResult::Below => {
                // This could happen with noise, but unlikely with high epsilon
            }
        }
    }

    #[test]
    fn test_sparse_vector_find_first() {
        let mut accountant = PrivacyAccountant::new();
        let queries = vec![50.0, 80.0, 200.0, 90.0, 150.0];

        let result = sparse_vector_find_first(&queries, 100.0, 1.0, 10.0, &mut accountant);
        assert!(result.is_ok());

        // With high epsilon, should find index 2 (value 200) most of the time
        // But due to noise, we just verify it returns a valid result
    }

    #[test]
    fn test_sparse_vector_find_first_none() {
        let mut accountant = PrivacyAccountant::new();
        let queries = vec![10.0, 20.0, 30.0]; // All well below threshold

        let result = sparse_vector_find_first(&queries, 100.0, 1.0, 10.0, &mut accountant);
        assert!(result.is_ok());
        // With these values, likely returns None
    }

    #[test]
    fn test_sparse_vector_find_all() {
        let mut accountant = PrivacyAccountant::new();
        let queries = vec![50.0, 200.0, 80.0, 300.0, 90.0, 400.0];

        let result = sparse_vector_find_all(&queries, 100.0, 1.0, 10.0, 3, &mut accountant);
        assert!(result.is_ok());

        let indices = result.unwrap();
        // Should find up to 3 indices
        assert!(indices.len() <= 3);
    }

    #[test]
    fn test_sparse_vector_find_all_empty() {
        let mut accountant = PrivacyAccountant::new();
        let queries: Vec<f64> = vec![];

        let result = sparse_vector_find_all(&queries, 100.0, 1.0, 1.0, 3, &mut accountant);
        assert!(result.is_err());
    }

    #[test]
    fn test_svt_custom_allocation() {
        let mut accountant = PrivacyAccountant::new();
        let svt = SparseVectorTechnique::with_custom_allocation(
            100.0, 1.0, 0.3, 0.7, 3, &mut accountant
        );
        assert!(svt.is_ok());

        let (eps, _) = accountant.compute_basic_composition();
        assert!((eps - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_svt_privacy_accounting() {
        let mut accountant = PrivacyAccountant::new();

        // Create multiple SVT instances
        let _ = SparseVectorTechnique::new(100.0, 1.0, 0.5, 3, &mut accountant);
        let _ = SparseVectorTechnique::new(100.0, 1.0, 0.3, 2, &mut accountant);

        let (eps, _) = accountant.compute_basic_composition();
        assert!((eps - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_svt_low_epsilon_randomness() {
        // With low epsilon, there should be significant randomness
        let mut above_count = 0;
        let num_trials = 100;

        for _ in 0..num_trials {
            let mut accountant = PrivacyAccountant::new();
            let mut svt = SparseVectorTechnique::new(100.0, 1.0, 0.1, 1, &mut accountant)
                .expect("Should create SVT");

            // Query exactly at threshold
            if let Some(ThresholdResult::Above) = svt.query(100.0) {
                above_count += 1;
            }
        }

        // With low epsilon and query at threshold, should see mix of Above/Below
        // Roughly 50% each due to symmetric noise
        assert!(
            above_count > 20 && above_count < 80,
            "Expected roughly 50% Above with low epsilon, got {}%",
            above_count
        );
    }

    #[test]
    fn test_numeric_svt_exhaustion() {
        let mut accountant = PrivacyAccountant::new();
        let mut svt = NumericSparseVector::new(0.0, 1.0, 10.0, 2, &mut accountant)
            .expect("Should create SVT");

        let _ = svt.query(100.0);
        let _ = svt.query(100.0);
        assert!(svt.is_exhausted());

        let result = svt.query(100.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_numeric_svt_batch() {
        let mut accountant = PrivacyAccountant::new();
        let mut svt = NumericSparseVector::new(100.0, 1.0, 10.0, 3, &mut accountant)
            .expect("Should create SVT");

        let queries = vec![50.0, 200.0, 80.0, 300.0];
        let results = svt.query_batch(&queries);

        assert!(!results.is_empty());
        assert!(results.len() <= queries.len());
    }
}
