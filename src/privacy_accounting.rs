// src/privacy_accounting.rs

use std::fmt;

/// Error type for privacy budget violations.
#[derive(Debug, Clone, PartialEq)]
pub enum PrivacyBudgetError {
    /// The requested epsilon would exceed the budget.
    EpsilonBudgetExceeded {
        requested: f64,
        available: f64,
        total_after: f64,
        budget: f64,
    },
    /// The requested delta would exceed the budget.
    DeltaBudgetExceeded {
        requested: f64,
        available: f64,
        total_after: f64,
        budget: f64,
    },
    /// Both epsilon and delta would exceed their budgets.
    BothBudgetsExceeded {
        epsilon_requested: f64,
        epsilon_available: f64,
        delta_requested: f64,
        delta_available: f64,
    },
    /// No budget has been set, but budget enforcement was requested.
    NoBudgetSet,
    /// Invalid parameters provided.
    InvalidParameters(String),
}

impl fmt::Display for PrivacyBudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrivacyBudgetError::EpsilonBudgetExceeded {
                requested,
                available,
                total_after,
                budget,
            } => write!(
                f,
                "Epsilon budget exceeded: requested ε={:.4} would result in total ε={:.4}, \
                 but budget is ε={:.4} (available: {:.4})",
                requested, total_after, budget, available
            ),
            PrivacyBudgetError::DeltaBudgetExceeded {
                requested,
                available,
                total_after,
                budget,
            } => write!(
                f,
                "Delta budget exceeded: requested δ={:.2e} would result in total δ={:.2e}, \
                 but budget is δ={:.2e} (available: {:.2e})",
                requested, total_after, budget, available
            ),
            PrivacyBudgetError::BothBudgetsExceeded {
                epsilon_requested,
                epsilon_available,
                delta_requested,
                delta_available,
            } => write!(
                f,
                "Both budgets exceeded: requested ε={:.4} (available: {:.4}), \
                 δ={:.2e} (available: {:.2e})",
                epsilon_requested, epsilon_available, delta_requested, delta_available
            ),
            PrivacyBudgetError::NoBudgetSet => {
                write!(f, "No privacy budget has been set for this accountant")
            }
            PrivacyBudgetError::InvalidParameters(msg) => {
                write!(f, "Invalid parameters: {}", msg)
            }
        }
    }
}

impl std::error::Error for PrivacyBudgetError {}

/// Composition method for computing total privacy loss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompositionMethod {
    /// Basic composition: ε_total = Σε_i, δ_total = Σδ_i
    /// Simple but loose bounds.
    Basic,

    /// Advanced composition (Dwork, Rothblum, Vadhan 2010):
    /// For k mechanisms each with (ε, δ)-DP:
    /// ε' = ε√(2k·ln(1/δ')) + k·ε(e^ε - 1)
    /// Provides tighter bounds for multiple queries.
    Advanced,

    /// Optimal advanced composition using numerical optimization.
    /// Provides the tightest bounds but is more computationally expensive.
    OptimalAdvanced,
}

/// Represents a single privacy query/mechanism invocation.
#[derive(Debug, Clone, Copy)]
pub struct PrivacyQuery {
    /// The epsilon (ε) consumed by this query.
    pub epsilon: f64,
    /// The delta (δ) consumed by this query.
    pub delta: f64,
}

/// Tracks cumulative privacy loss with support for multiple composition methods.
///
/// The `PrivacyAccountant` records individual privacy queries and can compute
/// the total privacy loss using different composition theorems. This allows
/// for tighter privacy bounds compared to naive composition.
///
/// # Example
///
/// ```rust
/// use differential_privacy::privacy_accounting::{PrivacyAccountant, CompositionMethod};
///
/// let mut accountant = PrivacyAccountant::new();
///
/// // Record several queries
/// accountant.update(0.1, 1e-6);
/// accountant.update(0.1, 1e-6);
/// accountant.update(0.1, 1e-6);
///
/// // Compare composition methods
/// let (basic_eps, basic_delta) = accountant.get_privacy_loss(CompositionMethod::Basic);
/// let (adv_eps, adv_delta) = accountant.get_privacy_loss_advanced(1e-5);
///
/// // Advanced composition typically gives lower epsilon!
/// println!("Basic: ε={}, δ={}", basic_eps, basic_delta);
/// println!("Advanced: ε={}, δ={}", adv_eps, adv_delta);
/// ```
#[derive(Debug, Clone)]
pub struct PrivacyAccountant {
    /// Budget limit for epsilon (if set).
    budget_epsilon: Option<f64>,
    /// Budget limit for delta (if set).
    budget_delta: Option<f64>,
    /// History of all privacy queries for advanced composition.
    queries: Vec<PrivacyQuery>,
}

impl Default for PrivacyAccountant {
    fn default() -> Self {
        Self::new()
    }
}

impl PrivacyAccountant {
    /// Creates a new `PrivacyAccountant` with no budget limits.
    pub fn new() -> Self {
        Self {
            budget_epsilon: None,
            budget_delta: None,
            queries: Vec::new(),
        }
    }

    /// Creates a new `PrivacyAccountant` with specified budget limits.
    ///
    /// # Arguments
    ///
    /// * `epsilon_budget` - Maximum allowed total epsilon.
    /// * `delta_budget` - Maximum allowed total delta.
    pub fn with_budget(epsilon_budget: f64, delta_budget: f64) -> Self {
        Self {
            budget_epsilon: Some(epsilon_budget),
            budget_delta: Some(delta_budget),
            queries: Vec::new(),
        }
    }

    /// Records a new privacy query/mechanism invocation.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - The ε consumed by the mechanism.
    /// * `delta` - The δ consumed by the mechanism.
    ///
    /// # Note
    ///
    /// This method does not check budget limits. Use `try_update` for budget enforcement.
    pub fn update(&mut self, epsilon: f64, delta: f64) {
        self.queries.push(PrivacyQuery { epsilon, delta });
    }

    /// Attempts to record a new privacy query, checking budget limits first.
    ///
    /// This method will fail if the query would cause the privacy budget to be exceeded.
    /// Uses basic composition for budget checking.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - The ε consumed by the mechanism.
    /// * `delta` - The δ consumed by the mechanism.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the query was recorded, or an error if budget would be exceeded.
    ///
    /// # Example
    ///
    /// ```rust
    /// use differential_privacy::privacy_accounting::PrivacyAccountant;
    ///
    /// let mut accountant = PrivacyAccountant::with_budget(1.0, 1e-5);
    ///
    /// // First query succeeds
    /// assert!(accountant.try_update(0.5, 1e-6).is_ok());
    ///
    /// // Second query that exceeds budget fails
    /// assert!(accountant.try_update(0.6, 1e-6).is_err());
    /// ```
    pub fn try_update(&mut self, epsilon: f64, delta: f64) -> Result<(), PrivacyBudgetError> {
        self.try_update_with_method(epsilon, delta, CompositionMethod::Basic)
    }

    /// Attempts to record a new privacy query using the specified composition method.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - The ε consumed by the mechanism.
    /// * `delta` - The δ consumed by the mechanism.
    /// * `method` - The composition method to use for budget checking.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the query was recorded, or an error if budget would be exceeded.
    pub fn try_update_with_method(
        &mut self,
        epsilon: f64,
        delta: f64,
        method: CompositionMethod,
    ) -> Result<(), PrivacyBudgetError> {
        // Check if we can afford this query
        self.can_afford_with_method(epsilon, delta, method)?;

        // Record the query
        self.queries.push(PrivacyQuery { epsilon, delta });
        Ok(())
    }

    /// Checks if a query with the given parameters can be made within budget.
    ///
    /// Uses basic composition for budget checking.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - The proposed ε for the query.
    /// * `delta` - The proposed δ for the query.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the query can be made, or an error describing why not.
    pub fn can_afford(&self, epsilon: f64, delta: f64) -> Result<(), PrivacyBudgetError> {
        self.can_afford_with_method(epsilon, delta, CompositionMethod::Basic)
    }

    /// Checks if a query can be made within budget using the specified composition method.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - The proposed ε for the query.
    /// * `delta` - The proposed δ for the query.
    /// * `method` - The composition method to use for budget calculation.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the query can be made, or an error describing why not.
    pub fn can_afford_with_method(
        &self,
        epsilon: f64,
        delta: f64,
        method: CompositionMethod,
    ) -> Result<(), PrivacyBudgetError> {
        // Validate parameters
        if epsilon < 0.0 {
            return Err(PrivacyBudgetError::InvalidParameters(
                "Epsilon must be non-negative".to_string(),
            ));
        }
        if delta < 0.0 {
            return Err(PrivacyBudgetError::InvalidParameters(
                "Delta must be non-negative".to_string(),
            ));
        }

        // If no budget is set, any query is allowed
        if self.budget_epsilon.is_none() && self.budget_delta.is_none() {
            return Ok(());
        }

        // Simulate adding the query
        let mut temp_queries = self.queries.clone();
        temp_queries.push(PrivacyQuery { epsilon, delta });

        // Create a temporary accountant to compute the new privacy loss
        let temp_accountant = PrivacyAccountant {
            budget_epsilon: self.budget_epsilon,
            budget_delta: self.budget_delta,
            queries: temp_queries,
        };

        let (new_epsilon, new_delta) = temp_accountant.get_privacy_loss(method);
        let (current_epsilon, current_delta) = self.get_privacy_loss(method);

        let epsilon_exceeded = self
            .budget_epsilon
            .map(|b| new_epsilon > b)
            .unwrap_or(false);

        let delta_exceeded = self
            .budget_delta
            .map(|b| new_delta > b)
            .unwrap_or(false);

        match (epsilon_exceeded, delta_exceeded) {
            (true, true) => Err(PrivacyBudgetError::BothBudgetsExceeded {
                epsilon_requested: epsilon,
                epsilon_available: self.budget_epsilon.unwrap_or(f64::INFINITY) - current_epsilon,
                delta_requested: delta,
                delta_available: self.budget_delta.unwrap_or(f64::INFINITY) - current_delta,
            }),
            (true, false) => Err(PrivacyBudgetError::EpsilonBudgetExceeded {
                requested: epsilon,
                available: self.budget_epsilon.unwrap() - current_epsilon,
                total_after: new_epsilon,
                budget: self.budget_epsilon.unwrap(),
            }),
            (false, true) => Err(PrivacyBudgetError::DeltaBudgetExceeded {
                requested: delta,
                available: self.budget_delta.unwrap() - current_delta,
                total_after: new_delta,
                budget: self.budget_delta.unwrap(),
            }),
            (false, false) => Ok(()),
        }
    }

    /// Returns whether a budget has been set for this accountant.
    pub fn has_budget(&self) -> bool {
        self.budget_epsilon.is_some() || self.budget_delta.is_some()
    }

    /// Returns the budget limits, if set.
    ///
    /// # Returns
    ///
    /// A tuple of (epsilon_budget, delta_budget), where each may be `None`.
    pub fn get_budget(&self) -> (Option<f64>, Option<f64>) {
        (self.budget_epsilon, self.budget_delta)
    }

    /// Sets new budget limits.
    ///
    /// # Arguments
    ///
    /// * `epsilon_budget` - New epsilon budget (use `None` to remove limit).
    /// * `delta_budget` - New delta budget (use `None` to remove limit).
    pub fn set_budget(&mut self, epsilon_budget: Option<f64>, delta_budget: Option<f64>) {
        self.budget_epsilon = epsilon_budget;
        self.budget_delta = delta_budget;
    }

    /// Returns the number of queries recorded.
    pub fn num_queries(&self) -> usize {
        self.queries.len()
    }

    /// Returns a reference to all recorded queries.
    pub fn get_queries(&self) -> &[PrivacyQuery] {
        &self.queries
    }

    /// Computes the total privacy loss using the specified composition method.
    ///
    /// # Arguments
    ///
    /// * `method` - The composition method to use.
    ///
    /// # Returns
    ///
    /// A tuple containing (total_epsilon, total_delta).
    ///
    /// # Note
    ///
    /// For `Advanced` and `OptimalAdvanced` methods, this uses a default
    /// delta' of 1e-6. Use `get_privacy_loss_advanced` for custom delta'.
    pub fn get_privacy_loss(&self, method: CompositionMethod) -> (f64, f64) {
        match method {
            CompositionMethod::Basic => self.compute_basic_composition(),
            CompositionMethod::Advanced => self.get_privacy_loss_advanced(1e-6),
            CompositionMethod::OptimalAdvanced => self.get_privacy_loss_optimal(1e-6),
        }
    }

    /// Computes privacy loss using basic composition (simple sum).
    ///
    /// # Returns
    ///
    /// A tuple containing (total_epsilon, total_delta).
    pub fn compute_basic_composition(&self) -> (f64, f64) {
        let total_epsilon: f64 = self.queries.iter().map(|q| q.epsilon).sum();
        let total_delta: f64 = self.queries.iter().map(|q| q.delta).sum();
        (total_epsilon, total_delta)
    }

    /// Computes privacy loss using advanced composition theorem.
    ///
    /// The advanced composition theorem (Dwork, Rothblum, Vadhan 2010) states that
    /// for k adaptive (ε, δ)-DP mechanisms, the composition is (ε', kδ + δ')-DP where:
    ///
    /// ε' = √(2k·ln(1/δ'))·ε + k·ε·(e^ε - 1)
    ///
    /// For heterogeneous epsilon values, we use the generalized form.
    ///
    /// # Arguments
    ///
    /// * `delta_prime` - Additional delta slack parameter (δ'). Smaller values
    ///                   give tighter epsilon but larger total delta.
    ///
    /// # Returns
    ///
    /// A tuple containing (total_epsilon, total_delta).
    pub fn get_privacy_loss_advanced(&self, delta_prime: f64) -> (f64, f64) {
        if self.queries.is_empty() {
            return (0.0, 0.0);
        }

        let sum_delta: f64 = self.queries.iter().map(|q| q.delta).sum();

        // For heterogeneous epsilons, we use the generalized advanced composition:
        // ε' = √(2·ln(1/δ')·Σε_i²) + Σε_i·(e^ε_i - 1)

        let sum_epsilon_squared: f64 = self.queries.iter().map(|q| q.epsilon.powi(2)).sum();

        let sum_epsilon_exp_term: f64 = self
            .queries
            .iter()
            .map(|q| q.epsilon * (q.epsilon.exp() - 1.0))
            .sum();

        // ε' = √(2·ln(1/δ')·Σε_i²) + Σε_i·(e^ε_i - 1)
        let epsilon_prime = (2.0 * (1.0 / delta_prime).ln() * sum_epsilon_squared).sqrt()
            + sum_epsilon_exp_term;

        // Total delta = Σδ_i + δ'
        let total_delta = sum_delta + delta_prime;

        // Fall back to basic composition if advanced gives worse bounds
        let (basic_epsilon, basic_delta) = self.compute_basic_composition();
        if epsilon_prime > basic_epsilon {
            return (basic_epsilon, basic_delta);
        }

        (epsilon_prime, total_delta)
    }

    /// Computes privacy loss using optimal advanced composition.
    ///
    /// This method finds the optimal δ' that minimizes the total privacy loss
    /// while respecting the given delta budget.
    ///
    /// # Arguments
    ///
    /// * `target_delta` - The target total delta budget.
    ///
    /// # Returns
    ///
    /// A tuple containing (total_epsilon, total_delta).
    pub fn get_privacy_loss_optimal(&self, target_delta: f64) -> (f64, f64) {
        if self.queries.is_empty() {
            return (0.0, 0.0);
        }

        let sum_delta: f64 = self.queries.iter().map(|q| q.delta).sum();

        // δ' must be positive and less than target_delta - sum_delta
        let max_delta_prime = target_delta - sum_delta;
        if max_delta_prime <= 0.0 {
            // Cannot achieve target delta with advanced composition
            return self.compute_basic_composition();
        }

        // Binary search for optimal delta_prime
        let mut best_epsilon = f64::INFINITY;
        let mut best_delta = target_delta;

        // Search over logarithmic scale for delta_prime
        let num_steps = 100;
        let log_min = (1e-15_f64).ln();
        let log_max = max_delta_prime.ln();

        for i in 0..num_steps {
            let t = i as f64 / (num_steps - 1) as f64;
            let log_delta_prime = log_min + t * (log_max - log_min);
            let delta_prime = log_delta_prime.exp();

            let (eps, del) = self.get_privacy_loss_advanced(delta_prime);
            if eps < best_epsilon && del <= target_delta {
                best_epsilon = eps;
                best_delta = del;
            }
        }

        if best_epsilon.is_infinite() {
            return self.compute_basic_composition();
        }

        (best_epsilon, best_delta)
    }

    /// Computes the privacy loss using Rényi Differential Privacy (RDP) composition.
    ///
    /// RDP provides even tighter composition for Gaussian mechanisms.
    /// This computes the RDP at order α and converts to (ε, δ)-DP.
    ///
    /// # Arguments
    ///
    /// * `alpha` - The Rényi divergence order (α > 1).
    /// * `target_delta` - The target delta for conversion to (ε, δ)-DP.
    ///
    /// # Returns
    ///
    /// A tuple containing (epsilon, delta) after RDP to DP conversion.
    pub fn get_privacy_loss_rdp(&self, alpha: f64, target_delta: f64) -> (f64, f64) {
        if self.queries.is_empty() || alpha <= 1.0 {
            return (0.0, 0.0);
        }

        // For Gaussian mechanism with (ε, δ)-DP parameters, the RDP guarantee at order α is:
        // ρ(α) ≈ α·ε² / 2 (approximation for small ε)
        //
        // RDP composes by addition: ρ_total(α) = Σρ_i(α)
        //
        // Convert RDP to (ε, δ)-DP: ε = ρ(α) + ln(1/δ)/(α-1)

        let total_rdp: f64 = self
            .queries
            .iter()
            .map(|q| alpha * q.epsilon.powi(2) / 2.0)
            .sum();

        // Convert RDP to (ε, δ)-DP
        let epsilon = total_rdp + (1.0 / target_delta).ln() / (alpha - 1.0);

        (epsilon, target_delta)
    }

    /// Finds the optimal α for RDP composition and converts to (ε, δ)-DP.
    ///
    /// # Arguments
    ///
    /// * `target_delta` - The target delta for the final (ε, δ)-DP guarantee.
    ///
    /// # Returns
    ///
    /// A tuple containing (epsilon, delta).
    pub fn get_privacy_loss_rdp_optimal(&self, target_delta: f64) -> (f64, f64) {
        if self.queries.is_empty() {
            return (0.0, 0.0);
        }

        let mut best_epsilon = f64::INFINITY;

        // Search for optimal α
        // Common range is [1.1, 100] with denser sampling near lower values
        let alphas: Vec<f64> = (0..100)
            .map(|i| 1.0 + 0.1 * (1.0 + i as f64 / 10.0).powi(2))
            .collect();

        for alpha in alphas {
            let (eps, _) = self.get_privacy_loss_rdp(alpha, target_delta);
            if eps < best_epsilon {
                best_epsilon = eps;
            }
        }

        if best_epsilon.is_infinite() {
            return self.compute_basic_composition();
        }

        (best_epsilon, target_delta)
    }

    /// Checks if the current privacy loss exceeds the budget.
    ///
    /// # Arguments
    ///
    /// * `method` - The composition method to use for computing privacy loss.
    ///
    /// # Returns
    ///
    /// `true` if the budget is exceeded, `false` otherwise.
    /// Returns `false` if no budget is set.
    pub fn is_budget_exceeded(&self, method: CompositionMethod) -> bool {
        let (epsilon, delta) = self.get_privacy_loss(method);

        let epsilon_exceeded = self
            .budget_epsilon
            .map(|b| epsilon > b)
            .unwrap_or(false);

        let delta_exceeded = self
            .budget_delta
            .map(|b| delta > b)
            .unwrap_or(false);

        epsilon_exceeded || delta_exceeded
    }

    /// Returns the remaining privacy budget.
    ///
    /// # Arguments
    ///
    /// * `method` - The composition method to use for computing current privacy loss.
    ///
    /// # Returns
    ///
    /// A tuple containing (remaining_epsilon, remaining_delta).
    /// Returns `None` values if no budget is set.
    pub fn get_remaining_budget(&self, method: CompositionMethod) -> (Option<f64>, Option<f64>) {
        let (epsilon, delta) = self.get_privacy_loss(method);

        let remaining_epsilon = self.budget_epsilon.map(|b| (b - epsilon).max(0.0));
        let remaining_delta = self.budget_delta.map(|b| (b - delta).max(0.0));

        (remaining_epsilon, remaining_delta)
    }

    /// Resets the accountant, clearing all recorded queries.
    pub fn reset(&mut self) {
        self.queries.clear();
    }

    /// Returns a summary comparing all composition methods.
    pub fn composition_summary(&self, target_delta: f64) -> CompositionSummary {
        let basic = self.compute_basic_composition();
        let advanced = self.get_privacy_loss_advanced(target_delta);
        let optimal = self.get_privacy_loss_optimal(target_delta);
        let rdp = self.get_privacy_loss_rdp_optimal(target_delta);

        CompositionSummary {
            num_queries: self.queries.len(),
            basic_epsilon: basic.0,
            basic_delta: basic.1,
            advanced_epsilon: advanced.0,
            advanced_delta: advanced.1,
            optimal_epsilon: optimal.0,
            optimal_delta: optimal.1,
            rdp_epsilon: rdp.0,
            rdp_delta: rdp.1,
        }
    }
}

/// Summary of privacy loss under different composition methods.
#[derive(Debug, Clone)]
pub struct CompositionSummary {
    /// Number of queries composed.
    pub num_queries: usize,
    /// Epsilon under basic composition.
    pub basic_epsilon: f64,
    /// Delta under basic composition.
    pub basic_delta: f64,
    /// Epsilon under advanced composition.
    pub advanced_epsilon: f64,
    /// Delta under advanced composition.
    pub advanced_delta: f64,
    /// Epsilon under optimal advanced composition.
    pub optimal_epsilon: f64,
    /// Delta under optimal advanced composition.
    pub optimal_delta: f64,
    /// Epsilon under RDP composition.
    pub rdp_epsilon: f64,
    /// Delta under RDP composition.
    pub rdp_delta: f64,
}

impl std::fmt::Display for CompositionSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Privacy Composition Summary ({} queries)", self.num_queries)?;
        writeln!(f, "----------------------------------------")?;
        writeln!(f, "Basic:    ε = {:.6}, δ = {:.2e}", self.basic_epsilon, self.basic_delta)?;
        writeln!(f, "Advanced: ε = {:.6}, δ = {:.2e}", self.advanced_epsilon, self.advanced_delta)?;
        writeln!(f, "Optimal:  ε = {:.6}, δ = {:.2e}", self.optimal_epsilon, self.optimal_delta)?;
        writeln!(f, "RDP:      ε = {:.6}, δ = {:.2e}", self.rdp_epsilon, self.rdp_delta)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_privacy_accountant_initialization() {
        let accountant = PrivacyAccountant::new();
        assert_eq!(accountant.num_queries(), 0);
        let (epsilon, delta) = accountant.compute_basic_composition();
        assert_eq!(epsilon, 0.0);
        assert_eq!(delta, 0.0);
    }

    #[test]
    fn test_privacy_accountant_with_budget() {
        let accountant = PrivacyAccountant::with_budget(1.0, 1e-5);
        assert_eq!(accountant.budget_epsilon, Some(1.0));
        assert_eq!(accountant.budget_delta, Some(1e-5));
    }

    #[test]
    fn test_basic_composition() {
        let mut accountant = PrivacyAccountant::new();
        accountant.update(0.5, 1e-6);
        accountant.update(0.3, 2e-6);
        accountant.update(0.2, 1e-6);

        let (epsilon, delta) = accountant.compute_basic_composition();
        assert!((epsilon - 1.0).abs() < 1e-10);
        assert!((delta - 4e-6).abs() < 1e-15);
    }

    #[test]
    fn test_advanced_composition_tighter_than_basic() {
        let mut accountant = PrivacyAccountant::new();

        // Add many small epsilon queries
        for _ in 0..100 {
            accountant.update(0.1, 0.0);
        }

        let (basic_epsilon, _) = accountant.compute_basic_composition();
        let (advanced_epsilon, _) = accountant.get_privacy_loss_advanced(1e-5);

        // Basic: 100 * 0.1 = 10.0
        assert!((basic_epsilon - 10.0).abs() < 1e-10);

        // Advanced should be significantly less
        assert!(
            advanced_epsilon < basic_epsilon,
            "Advanced ({}) should be less than basic ({})",
            advanced_epsilon,
            basic_epsilon
        );

        // For 100 queries with ε=0.1, advanced composition should give roughly:
        // ε' ≈ √(2·100·ln(10^5))·0.1 + 100·0.1·(e^0.1 - 1)
        // ε' ≈ √(2·100·11.5)·0.1 + 100·0.1·0.105
        // ε' ≈ 4.8 + 1.05 ≈ 5.85
        assert!(
            advanced_epsilon < 7.0,
            "Advanced epsilon {} should be around 5-6",
            advanced_epsilon
        );
    }

    #[test]
    fn test_advanced_composition_heterogeneous() {
        let mut accountant = PrivacyAccountant::new();

        // Different epsilon values
        accountant.update(0.5, 1e-6);
        accountant.update(0.1, 1e-6);
        accountant.update(0.3, 1e-6);

        let (basic_epsilon, basic_delta) = accountant.compute_basic_composition();
        let (advanced_epsilon, advanced_delta) = accountant.get_privacy_loss_advanced(1e-5);

        assert!((basic_epsilon - 0.9).abs() < 1e-10);
        assert!((basic_delta - 3e-6).abs() < 1e-15);

        // Advanced should be <= basic (or fall back to basic if worse)
        assert!(advanced_epsilon <= basic_epsilon + 1e-10);
    }

    #[test]
    fn test_optimal_composition() {
        let mut accountant = PrivacyAccountant::new();

        for _ in 0..50 {
            accountant.update(0.1, 1e-8);
        }

        let (basic_epsilon, _) = accountant.compute_basic_composition();
        let (optimal_epsilon, optimal_delta) = accountant.get_privacy_loss_optimal(1e-5);

        assert!(optimal_epsilon < basic_epsilon);
        assert!(optimal_delta <= 1e-5);
    }

    #[test]
    fn test_rdp_composition() {
        let mut accountant = PrivacyAccountant::new();

        for _ in 0..100 {
            accountant.update(0.1, 0.0);
        }

        let (rdp_epsilon, rdp_delta) = accountant.get_privacy_loss_rdp(10.0, 1e-5);
        let (basic_epsilon, _) = accountant.compute_basic_composition();

        // RDP should give a finite epsilon
        assert!(rdp_epsilon.is_finite());
        assert!(rdp_epsilon > 0.0);

        // For many queries, RDP often gives better bounds
        assert!(rdp_epsilon < basic_epsilon * 2.0); // Sanity check
    }

    #[test]
    fn test_rdp_optimal() {
        let mut accountant = PrivacyAccountant::new();

        for _ in 0..20 {
            accountant.update(0.2, 0.0);
        }

        let (rdp_optimal_epsilon, _) = accountant.get_privacy_loss_rdp_optimal(1e-5);

        assert!(rdp_optimal_epsilon.is_finite());
        assert!(rdp_optimal_epsilon > 0.0);
    }

    #[test]
    fn test_budget_enforcement() {
        let mut accountant = PrivacyAccountant::with_budget(1.0, 1e-5);

        accountant.update(0.3, 1e-6);
        assert!(!accountant.is_budget_exceeded(CompositionMethod::Basic));

        accountant.update(0.3, 1e-6);
        assert!(!accountant.is_budget_exceeded(CompositionMethod::Basic));

        accountant.update(0.5, 1e-6);
        assert!(accountant.is_budget_exceeded(CompositionMethod::Basic));
    }

    #[test]
    fn test_remaining_budget() {
        let mut accountant = PrivacyAccountant::with_budget(1.0, 1e-5);

        accountant.update(0.3, 1e-6);
        let (remaining_eps, remaining_delta) =
            accountant.get_remaining_budget(CompositionMethod::Basic);

        assert!((remaining_eps.unwrap() - 0.7).abs() < 1e-10);
        assert!((remaining_delta.unwrap() - (1e-5 - 1e-6)).abs() < 1e-15);
    }

    #[test]
    fn test_composition_summary() {
        let mut accountant = PrivacyAccountant::new();

        for _ in 0..20 {
            accountant.update(0.1, 1e-7);
        }

        let summary = accountant.composition_summary(1e-5);

        assert_eq!(summary.num_queries, 20);
        assert!((summary.basic_epsilon - 2.0).abs() < 1e-10);
        assert!(summary.advanced_epsilon <= summary.basic_epsilon);
        assert!(summary.optimal_epsilon <= summary.basic_epsilon);
    }

    #[test]
    fn test_reset() {
        let mut accountant = PrivacyAccountant::new();
        accountant.update(0.5, 1e-6);
        accountant.update(0.5, 1e-6);

        assert_eq!(accountant.num_queries(), 2);

        accountant.reset();

        assert_eq!(accountant.num_queries(), 0);
        let (epsilon, delta) = accountant.compute_basic_composition();
        assert_eq!(epsilon, 0.0);
        assert_eq!(delta, 0.0);
    }

    #[test]
    fn test_empty_accountant() {
        let accountant = PrivacyAccountant::new();

        let (basic_eps, basic_delta) = accountant.compute_basic_composition();
        assert_eq!(basic_eps, 0.0);
        assert_eq!(basic_delta, 0.0);

        let (adv_eps, adv_delta) = accountant.get_privacy_loss_advanced(1e-5);
        assert_eq!(adv_eps, 0.0);
        assert_eq!(adv_delta, 0.0);

        let (opt_eps, opt_delta) = accountant.get_privacy_loss_optimal(1e-5);
        assert_eq!(opt_eps, 0.0);
        assert_eq!(opt_delta, 0.0);
    }

    #[test]
    fn test_single_query_composition() {
        let mut accountant = PrivacyAccountant::new();
        accountant.update(0.5, 1e-6);

        let (basic_eps, basic_delta) = accountant.compute_basic_composition();
        let (adv_eps, _) = accountant.get_privacy_loss_advanced(1e-6);

        // For a single query, basic is optimal
        assert!((basic_eps - 0.5).abs() < 1e-10);
        assert!((basic_delta - 1e-6).abs() < 1e-15);

        // Advanced should fall back to basic for single query
        assert!(adv_eps <= basic_eps + 1e-10);
    }

    #[test]
    fn test_get_queries() {
        let mut accountant = PrivacyAccountant::new();
        accountant.update(0.1, 1e-6);
        accountant.update(0.2, 2e-6);

        let queries = accountant.get_queries();
        assert_eq!(queries.len(), 2);
        assert!((queries[0].epsilon - 0.1).abs() < 1e-10);
        assert!((queries[1].epsilon - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_composition_improvement_ratio() {
        // Demonstrate the improvement from advanced composition
        let mut accountant = PrivacyAccountant::new();

        // 1000 queries with small epsilon
        for _ in 0..1000 {
            accountant.update(0.01, 0.0);
        }

        let (basic_epsilon, _) = accountant.compute_basic_composition();
        let (advanced_epsilon, _) = accountant.get_privacy_loss_advanced(1e-5);

        // Basic: 1000 * 0.01 = 10.0
        assert!((basic_epsilon - 10.0).abs() < 1e-10);

        // Advanced should give significant improvement
        let improvement_ratio = basic_epsilon / advanced_epsilon;
        assert!(
            improvement_ratio > 2.0,
            "Expected >2x improvement, got {:.2}x",
            improvement_ratio
        );
    }

    #[test]
    fn test_try_update_success() {
        let mut accountant = PrivacyAccountant::with_budget(1.0, 1e-5);

        // Should succeed
        let result = accountant.try_update(0.3, 1e-6);
        assert!(result.is_ok());
        assert_eq!(accountant.num_queries(), 1);

        // Should succeed again
        let result = accountant.try_update(0.3, 1e-6);
        assert!(result.is_ok());
        assert_eq!(accountant.num_queries(), 2);
    }

    #[test]
    fn test_try_update_exceeds_epsilon() {
        let mut accountant = PrivacyAccountant::with_budget(1.0, 1e-5);

        // Use up most of the budget
        accountant.try_update(0.6, 0.0).unwrap();

        // This should fail
        let result = accountant.try_update(0.5, 0.0);
        assert!(result.is_err());

        match result {
            Err(PrivacyBudgetError::EpsilonBudgetExceeded { requested, budget, .. }) => {
                assert!((requested - 0.5).abs() < 1e-10);
                assert!((budget - 1.0).abs() < 1e-10);
            }
            _ => panic!("Expected EpsilonBudgetExceeded error"),
        }

        // Query should not have been recorded
        assert_eq!(accountant.num_queries(), 1);
    }

    #[test]
    fn test_try_update_exceeds_delta() {
        let mut accountant = PrivacyAccountant::with_budget(10.0, 1e-5);

        // This should fail due to delta
        let result = accountant.try_update(0.5, 1e-4);
        assert!(result.is_err());

        match result {
            Err(PrivacyBudgetError::DeltaBudgetExceeded { requested, budget, .. }) => {
                assert!((requested - 1e-4).abs() < 1e-15);
                assert!((budget - 1e-5).abs() < 1e-15);
            }
            _ => panic!("Expected DeltaBudgetExceeded error"),
        }
    }

    #[test]
    fn test_try_update_no_budget() {
        let mut accountant = PrivacyAccountant::new();

        // Should always succeed without budget
        let result = accountant.try_update(100.0, 1.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_can_afford_success() {
        let accountant = PrivacyAccountant::with_budget(1.0, 1e-5);

        let result = accountant.can_afford(0.5, 1e-6);
        assert!(result.is_ok());
    }

    #[test]
    fn test_can_afford_failure() {
        let mut accountant = PrivacyAccountant::with_budget(1.0, 1e-5);
        accountant.update(0.6, 0.0);

        let result = accountant.can_afford(0.5, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_can_afford_invalid_params() {
        let accountant = PrivacyAccountant::with_budget(1.0, 1e-5);

        // Negative epsilon
        let result = accountant.can_afford(-0.5, 0.0);
        assert!(matches!(result, Err(PrivacyBudgetError::InvalidParameters(_))));

        // Negative delta
        let result = accountant.can_afford(0.5, -1e-6);
        assert!(matches!(result, Err(PrivacyBudgetError::InvalidParameters(_))));
    }

    #[test]
    fn test_has_budget() {
        let accountant1 = PrivacyAccountant::new();
        assert!(!accountant1.has_budget());

        let accountant2 = PrivacyAccountant::with_budget(1.0, 1e-5);
        assert!(accountant2.has_budget());
    }

    #[test]
    fn test_get_budget() {
        let accountant = PrivacyAccountant::with_budget(1.0, 1e-5);
        let (eps, delta) = accountant.get_budget();
        assert_eq!(eps, Some(1.0));
        assert_eq!(delta, Some(1e-5));
    }

    #[test]
    fn test_set_budget() {
        let mut accountant = PrivacyAccountant::new();
        assert!(!accountant.has_budget());

        accountant.set_budget(Some(2.0), Some(1e-6));
        assert!(accountant.has_budget());

        let (eps, delta) = accountant.get_budget();
        assert_eq!(eps, Some(2.0));
        assert_eq!(delta, Some(1e-6));
    }

    #[test]
    fn test_try_update_with_advanced_composition() {
        let mut accountant = PrivacyAccountant::with_budget(10.0, 1e-4);

        // Add many small queries - should succeed with advanced composition
        for _ in 0..50 {
            let result = accountant.try_update_with_method(0.1, 0.0, CompositionMethod::Advanced);
            assert!(result.is_ok(), "Query should succeed with advanced composition");
        }

        // Basic composition would give 5.0, but we used 50 queries
        // Advanced composition should give a lower value for many queries
        let (eps, _) = accountant.get_privacy_loss(CompositionMethod::Advanced);
        let (basic_eps, _) = accountant.compute_basic_composition();

        // Advanced may equal basic for small numbers of queries, but should never be worse
        assert!(eps <= basic_eps, "Advanced should not exceed basic: {} > {}", eps, basic_eps);
    }

    #[test]
    fn test_privacy_budget_error_display() {
        let err = PrivacyBudgetError::EpsilonBudgetExceeded {
            requested: 0.5,
            available: 0.3,
            total_after: 1.2,
            budget: 1.0,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Epsilon budget exceeded"));
        assert!(msg.contains("0.5"));
    }

    #[test]
    fn test_both_budgets_exceeded() {
        let mut accountant = PrivacyAccountant::with_budget(0.5, 1e-6);

        let result = accountant.try_update(1.0, 1e-5);
        assert!(matches!(result, Err(PrivacyBudgetError::BothBudgetsExceeded { .. })));
    }
}
