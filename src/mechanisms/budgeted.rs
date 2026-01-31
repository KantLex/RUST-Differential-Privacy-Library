// src/mechanisms/budgeted.rs

//! Budget-enforcing wrappers for differential privacy mechanisms.
//!
//! This module provides versions of the standard mechanisms that automatically
//! check and enforce privacy budget limits before execution.

use crate::privacy_accounting::{CompositionMethod, PrivacyAccountant, PrivacyBudgetError};

/// Error type for budgeted mechanism operations.
#[derive(Debug)]
pub enum BudgetedMechanismError {
    /// The privacy budget would be exceeded by this operation.
    BudgetExceeded(PrivacyBudgetError),
    /// The mechanism parameters are invalid.
    InvalidParameters(&'static str),
}

impl std::fmt::Display for BudgetedMechanismError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BudgetedMechanismError::BudgetExceeded(e) => write!(f, "Budget exceeded: {}", e),
            BudgetedMechanismError::InvalidParameters(msg) => {
                write!(f, "Invalid parameters: {}", msg)
            }
        }
    }
}

impl std::error::Error for BudgetedMechanismError {}

impl From<PrivacyBudgetError> for BudgetedMechanismError {
    fn from(err: PrivacyBudgetError) -> Self {
        BudgetedMechanismError::BudgetExceeded(err)
    }
}

/// Adds Laplace noise to a value with budget enforcement.
///
/// This function checks the privacy budget before executing and will fail
/// if the budget would be exceeded.
///
/// # Arguments
///
/// * `value` - The original numeric value.
/// * `sensitivity` - The sensitivity of the query.
/// * `epsilon` - The privacy budget for this query.
/// * `accountant` - The privacy accountant to check and update.
///
/// # Returns
///
/// `Ok(noisy_value)` if successful, or an error if budget exceeded or invalid parameters.
///
/// # Example
///
/// ```rust
/// use differential_privacy::mechanisms::budgeted::laplace_mechanism_budgeted;
/// use differential_privacy::privacy_accounting::PrivacyAccountant;
///
/// let mut accountant = PrivacyAccountant::with_budget(1.0, 1e-5);
///
/// // First query succeeds
/// let result = laplace_mechanism_budgeted(100.0, 1.0, 0.5, &mut accountant);
/// assert!(result.is_ok());
///
/// // Query that would exceed budget fails
/// let result = laplace_mechanism_budgeted(100.0, 1.0, 0.6, &mut accountant);
/// assert!(result.is_err());
/// ```
pub fn laplace_mechanism_budgeted(
    value: f64,
    sensitivity: f64,
    epsilon: f64,
    accountant: &mut PrivacyAccountant,
) -> Result<f64, BudgetedMechanismError> {
    // Validate parameters
    if epsilon <= 0.0 {
        return Err(BudgetedMechanismError::InvalidParameters(
            "Epsilon must be positive.",
        ));
    }
    if sensitivity < 0.0 {
        return Err(BudgetedMechanismError::InvalidParameters(
            "Sensitivity must be non-negative.",
        ));
    }

    // Check budget (Laplace mechanism has delta = 0)
    accountant.try_update(epsilon, 0.0)?;

    // Remove the query we just added (we'll use the real mechanism to add it)
    // Actually, let's just compute the noise here since we already updated
    let scale = sensitivity / epsilon;
    let noise = sample_laplace(scale);

    Ok(value + noise)
}

/// Adds Gaussian noise to a value with budget enforcement.
///
/// This function checks the privacy budget before executing and will fail
/// if the budget would be exceeded.
///
/// # Arguments
///
/// * `value` - The original numeric value.
/// * `sensitivity` - The L2 sensitivity of the query.
/// * `epsilon` - The privacy budget for this query.
/// * `delta` - The delta parameter for (ε, δ)-DP.
/// * `accountant` - The privacy accountant to check and update.
///
/// # Returns
///
/// `Ok(noisy_value)` if successful, or an error if budget exceeded or invalid parameters.
pub fn gaussian_mechanism_budgeted(
    value: f64,
    sensitivity: f64,
    epsilon: f64,
    delta: f64,
    accountant: &mut PrivacyAccountant,
) -> Result<f64, BudgetedMechanismError> {
    // Validate parameters
    if epsilon <= 0.0 {
        return Err(BudgetedMechanismError::InvalidParameters(
            "Epsilon must be positive.",
        ));
    }
    if sensitivity < 0.0 {
        return Err(BudgetedMechanismError::InvalidParameters(
            "Sensitivity must be non-negative.",
        ));
    }
    if delta <= 0.0 || delta >= 1.0 {
        return Err(BudgetedMechanismError::InvalidParameters(
            "Delta must be in the range (0, 1).",
        ));
    }

    // Check budget
    accountant.try_update(epsilon, delta)?;

    // Calculate sigma and add noise
    let sigma = sensitivity * (2.0 * (1.25_f64 / delta).ln()).sqrt() / epsilon;

    if sigma == 0.0 {
        return Ok(value);
    }

    let noise = sample_gaussian(sigma);
    Ok(value + noise)
}

/// Selects an item using the exponential mechanism with budget enforcement.
///
/// # Arguments
///
/// * `utilities` - Utility scores for each candidate.
/// * `sensitivity` - The sensitivity of the utility function.
/// * `epsilon` - The privacy budget for this query.
/// * `accountant` - The privacy accountant to check and update.
///
/// # Returns
///
/// `Ok(index)` of the selected item, or an error if budget exceeded or invalid parameters.
pub fn exponential_mechanism_budgeted(
    utilities: &[f64],
    sensitivity: f64,
    epsilon: f64,
    accountant: &mut PrivacyAccountant,
) -> Result<usize, BudgetedMechanismError> {
    // Validate parameters
    if utilities.is_empty() {
        return Err(BudgetedMechanismError::InvalidParameters(
            "Utilities slice must not be empty.",
        ));
    }
    if epsilon <= 0.0 {
        return Err(BudgetedMechanismError::InvalidParameters(
            "Epsilon must be positive.",
        ));
    }
    if sensitivity <= 0.0 {
        return Err(BudgetedMechanismError::InvalidParameters(
            "Sensitivity must be positive.",
        ));
    }

    for &u in utilities {
        if u.is_nan() || u.is_infinite() {
            return Err(BudgetedMechanismError::InvalidParameters(
                "Utilities must be finite numbers.",
            ));
        }
    }

    // Check budget (Exponential mechanism has delta = 0)
    accountant.try_update(epsilon, 0.0)?;

    // Perform selection
    let scaling_factor = epsilon / (2.0 * sensitivity);
    let max_utility = utilities
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let weights: Vec<f64> = utilities
        .iter()
        .map(|&u| ((u - max_utility) * scaling_factor).exp())
        .collect();

    let total_weight: f64 = weights.iter().sum();
    sample_from_weights(&weights, total_weight)
        .map_err(|_| BudgetedMechanismError::InvalidParameters("Failed to sample from weights."))
}

/// Reports the noisy max with budget enforcement.
///
/// # Arguments
///
/// * `counts` - The count values.
/// * `sensitivity` - The sensitivity of each count.
/// * `epsilon` - The privacy budget for this query.
/// * `accountant` - The privacy accountant to check and update.
///
/// # Returns
///
/// `Ok((index, noisy_max))` or an error if budget exceeded or invalid parameters.
pub fn report_noisy_max_budgeted(
    counts: &[f64],
    sensitivity: f64,
    epsilon: f64,
    accountant: &mut PrivacyAccountant,
) -> Result<(usize, f64), BudgetedMechanismError> {
    // Validate parameters
    if counts.is_empty() {
        return Err(BudgetedMechanismError::InvalidParameters(
            "Counts slice must not be empty.",
        ));
    }
    if epsilon <= 0.0 {
        return Err(BudgetedMechanismError::InvalidParameters(
            "Epsilon must be positive.",
        ));
    }
    if sensitivity <= 0.0 {
        return Err(BudgetedMechanismError::InvalidParameters(
            "Sensitivity must be positive.",
        ));
    }

    for &c in counts {
        if c.is_nan() || c.is_infinite() {
            return Err(BudgetedMechanismError::InvalidParameters(
                "Counts must be finite numbers.",
            ));
        }
    }

    // Check budget
    accountant.try_update(epsilon, 0.0)?;

    // Find noisy max
    let scale = sensitivity / epsilon;
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

    Ok((max_index, max_noisy_value))
}

/// Configuration for budgeted mechanisms.
#[derive(Debug, Clone)]
pub struct BudgetedMechanismConfig {
    /// The composition method to use for budget checking.
    pub composition_method: CompositionMethod,
    /// Whether to allow operations that would exceed the budget by a small margin.
    pub allow_epsilon_tolerance: Option<f64>,
}

impl Default for BudgetedMechanismConfig {
    fn default() -> Self {
        Self {
            composition_method: CompositionMethod::Basic,
            allow_epsilon_tolerance: None,
        }
    }
}

/// A wrapper around PrivacyAccountant that provides automatic budget enforcement.
///
/// All mechanism calls through this wrapper will automatically check and enforce
/// the privacy budget.
///
/// # Example
///
/// ```rust
/// use differential_privacy::mechanisms::budgeted::BudgetedAccountant;
///
/// let mut accountant = BudgetedAccountant::new(1.0, 1e-5);
///
/// // Run queries with automatic budget checking
/// let result1 = accountant.laplace(100.0, 1.0, 0.3);
/// assert!(result1.is_ok());
///
/// let result2 = accountant.laplace(100.0, 1.0, 0.3);
/// assert!(result2.is_ok());
///
/// // This would exceed the budget
/// let result3 = accountant.laplace(100.0, 1.0, 0.5);
/// assert!(result3.is_err());
///
/// // Check remaining budget
/// println!("Remaining: {:?}", accountant.remaining_budget());
/// ```
#[derive(Debug, Clone)]
pub struct BudgetedAccountant {
    accountant: PrivacyAccountant,
    config: BudgetedMechanismConfig,
}

impl BudgetedAccountant {
    /// Creates a new BudgetedAccountant with the specified budget.
    pub fn new(epsilon_budget: f64, delta_budget: f64) -> Self {
        Self {
            accountant: PrivacyAccountant::with_budget(epsilon_budget, delta_budget),
            config: BudgetedMechanismConfig::default(),
        }
    }

    /// Creates a new BudgetedAccountant with custom configuration.
    pub fn with_config(
        epsilon_budget: f64,
        delta_budget: f64,
        config: BudgetedMechanismConfig,
    ) -> Self {
        Self {
            accountant: PrivacyAccountant::with_budget(epsilon_budget, delta_budget),
            config,
        }
    }

    /// Returns the remaining privacy budget.
    pub fn remaining_budget(&self) -> (Option<f64>, Option<f64>) {
        self.accountant
            .get_remaining_budget(self.config.composition_method)
    }

    /// Returns the current privacy loss.
    pub fn current_loss(&self) -> (f64, f64) {
        self.accountant.get_privacy_loss(self.config.composition_method)
    }

    /// Returns the number of queries executed.
    pub fn num_queries(&self) -> usize {
        self.accountant.num_queries()
    }

    /// Checks if a query with the given parameters can be afforded.
    pub fn can_afford(&self, epsilon: f64, delta: f64) -> Result<(), PrivacyBudgetError> {
        self.accountant
            .can_afford_with_method(epsilon, delta, self.config.composition_method)
    }

    /// Adds Laplace noise to a value.
    pub fn laplace(
        &mut self,
        value: f64,
        sensitivity: f64,
        epsilon: f64,
    ) -> Result<f64, BudgetedMechanismError> {
        laplace_mechanism_budgeted(value, sensitivity, epsilon, &mut self.accountant)
    }

    /// Adds Gaussian noise to a value.
    pub fn gaussian(
        &mut self,
        value: f64,
        sensitivity: f64,
        epsilon: f64,
        delta: f64,
    ) -> Result<f64, BudgetedMechanismError> {
        gaussian_mechanism_budgeted(value, sensitivity, epsilon, delta, &mut self.accountant)
    }

    /// Selects an item using the exponential mechanism.
    pub fn exponential(
        &mut self,
        utilities: &[f64],
        sensitivity: f64,
        epsilon: f64,
    ) -> Result<usize, BudgetedMechanismError> {
        exponential_mechanism_budgeted(utilities, sensitivity, epsilon, &mut self.accountant)
    }

    /// Reports the noisy max.
    pub fn report_noisy_max(
        &mut self,
        counts: &[f64],
        sensitivity: f64,
        epsilon: f64,
    ) -> Result<(usize, f64), BudgetedMechanismError> {
        report_noisy_max_budgeted(counts, sensitivity, epsilon, &mut self.accountant)
    }

    /// Returns a reference to the underlying accountant.
    pub fn accountant(&self) -> &PrivacyAccountant {
        &self.accountant
    }

    /// Resets the accountant, clearing all recorded queries.
    pub fn reset(&mut self) {
        self.accountant.reset();
    }
}

// Helper functions

fn sample_laplace(scale: f64) -> f64 {
    use rand::Rng;
    let uniform: f64 = rand::thread_rng().gen::<f64>() - 0.5;
    -(scale) * uniform.signum() * (1.0 - 2.0 * uniform.abs()).ln()
}

fn sample_gaussian(sigma: f64) -> f64 {
    use rand_distr::{Distribution, Normal};
    let normal = Normal::new(0.0, sigma).expect("Invalid sigma");
    normal.sample(&mut rand::thread_rng())
}

fn sample_from_weights(weights: &[f64], total_weight: f64) -> Result<usize, ()> {
    use rand::Rng;
    let sample: f64 = rand::thread_rng().gen::<f64>() * total_weight;
    let mut cumulative = 0.0;
    for (i, &weight) in weights.iter().enumerate() {
        cumulative += weight;
        if sample <= cumulative {
            return Ok(i);
        }
    }
    Ok(weights.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplace_budgeted_success() {
        let mut accountant = PrivacyAccountant::with_budget(1.0, 1e-5);
        let result = laplace_mechanism_budgeted(100.0, 1.0, 0.5, &mut accountant);
        assert!(result.is_ok());
        assert_eq!(accountant.num_queries(), 1);
    }

    #[test]
    fn test_laplace_budgeted_exceeds_budget() {
        let mut accountant = PrivacyAccountant::with_budget(1.0, 1e-5);

        // First query succeeds
        let result = laplace_mechanism_budgeted(100.0, 1.0, 0.6, &mut accountant);
        assert!(result.is_ok());

        // Second query exceeds budget
        let result = laplace_mechanism_budgeted(100.0, 1.0, 0.5, &mut accountant);
        assert!(result.is_err());
        assert_eq!(accountant.num_queries(), 1); // Should not have been recorded
    }

    #[test]
    fn test_gaussian_budgeted_success() {
        let mut accountant = PrivacyAccountant::with_budget(1.0, 1e-4);
        let result = gaussian_mechanism_budgeted(100.0, 1.0, 0.5, 1e-5, &mut accountant);
        assert!(result.is_ok());
        assert_eq!(accountant.num_queries(), 1);
    }

    #[test]
    fn test_gaussian_budgeted_exceeds_delta() {
        let mut accountant = PrivacyAccountant::with_budget(10.0, 1e-5);

        // Query with delta that would exceed budget
        let result = gaussian_mechanism_budgeted(100.0, 1.0, 0.5, 1e-4, &mut accountant);
        assert!(result.is_err());
        assert_eq!(accountant.num_queries(), 0);
    }

    #[test]
    fn test_exponential_budgeted_success() {
        let mut accountant = PrivacyAccountant::with_budget(1.0, 1e-5);
        let utilities = vec![10.0, 20.0, 15.0];
        let result = exponential_mechanism_budgeted(&utilities, 1.0, 0.5, &mut accountant);
        assert!(result.is_ok());
        assert!(result.unwrap() < utilities.len());
    }

    #[test]
    fn test_report_noisy_max_budgeted_success() {
        let mut accountant = PrivacyAccountant::with_budget(1.0, 1e-5);
        let counts = vec![100.0, 200.0, 150.0];
        let result = report_noisy_max_budgeted(&counts, 1.0, 0.5, &mut accountant);
        assert!(result.is_ok());
    }

    #[test]
    fn test_budgeted_accountant_tracks_usage() {
        let mut accountant = BudgetedAccountant::new(1.0, 1e-5);

        // Run several queries
        accountant.laplace(100.0, 1.0, 0.2).unwrap();
        accountant.laplace(100.0, 1.0, 0.2).unwrap();
        accountant.laplace(100.0, 1.0, 0.2).unwrap();

        assert_eq!(accountant.num_queries(), 3);

        let (remaining_eps, _) = accountant.remaining_budget();
        assert!((remaining_eps.unwrap() - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_budgeted_accountant_enforces_limit() {
        let mut accountant = BudgetedAccountant::new(1.0, 1e-5);

        // Use up most of the budget
        accountant.laplace(100.0, 1.0, 0.4).unwrap();
        accountant.laplace(100.0, 1.0, 0.4).unwrap();

        // This should fail
        let result = accountant.laplace(100.0, 1.0, 0.3);
        assert!(result.is_err());

        // Query count should not increase
        assert_eq!(accountant.num_queries(), 2);
    }

    #[test]
    fn test_budgeted_accountant_can_afford() {
        let mut accountant = BudgetedAccountant::new(1.0, 1e-5);

        // Should be able to afford this
        assert!(accountant.can_afford(0.5, 0.0).is_ok());

        // Use some budget
        accountant.laplace(100.0, 1.0, 0.6).unwrap();

        // Should not be able to afford this now
        assert!(accountant.can_afford(0.5, 0.0).is_err());
    }

    #[test]
    fn test_budgeted_accountant_reset() {
        let mut accountant = BudgetedAccountant::new(1.0, 1e-5);

        accountant.laplace(100.0, 1.0, 0.5).unwrap();
        accountant.laplace(100.0, 1.0, 0.4).unwrap();

        assert_eq!(accountant.num_queries(), 2);

        accountant.reset();

        assert_eq!(accountant.num_queries(), 0);
        let (remaining_eps, _) = accountant.remaining_budget();
        assert!((remaining_eps.unwrap() - 1.0).abs() < 1e-10);
    }
}
