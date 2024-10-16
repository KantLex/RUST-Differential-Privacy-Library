// src/privacy_accounting.rs

/// Tracks cumulative privacy loss using basic composition.
///
/// The `PrivacyAccountant` accumulates the privacy parameters (epsilon and delta)
/// to provide an overall privacy guarantee for multiple queries or operations.
pub struct PrivacyAccountant {
    epsilon: f64,
    delta: f64,
    total_epsilon: f64,
    total_delta: f64,
}

impl PrivacyAccountant {
    /// Creates a new `PrivacyAccountant` with initial privacy parameters.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - The privacy budget ε.
    /// * `delta` - The privacy parameter δ.
    pub fn new(epsilon: f64, delta: f64) -> Self {
        Self {
            epsilon,
            delta,
            total_epsilon: 0.0,
            total_delta: 0.0,
        }
    }

    /// Updates the cumulative privacy loss based on a new mechanism's parameters.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - The ε of the new mechanism.
    /// * `delta` - The δ of the new mechanism.
    pub fn update(&mut self, epsilon: f64, delta: f64) {
        self.total_epsilon += epsilon;
        self.total_delta += delta;
    }

    /// Retrieves the total accumulated privacy loss.
    ///
    /// # Returns
    ///
    /// A tuple containing (total_epsilon, total_delta).
    pub fn get_privacy_loss(&self) -> (f64, f64) {
        (self.total_epsilon, self.total_delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_privacy_accountant_initialization() {
        let accountant = PrivacyAccountant::new(1.0, 1e-5);
        assert_eq!(accountant.total_epsilon, 0.0);
        assert_eq!(accountant.total_delta, 0.0);
    }

    #[test]
    fn test_privacy_accountant_update() {
        let mut accountant = PrivacyAccountant::new(0.0, 0.0);
        accountant.update(0.5, 1e-6);
        accountant.update(0.3, 2e-6);
        assert_eq!(accountant.total_epsilon, 0.8);
        assert_eq!(accountant.total_delta, 3e-6);
    }
}