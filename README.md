# Differential Privacy Library in Rust

A robust, performant, and easy-to-use toolkit for incorporating privacy-preserving techniques into data analysis pipelines.

## Features

* **Epsilon-Delta Differential Privacy:** Supports both ε and (ε, δ)-differential privacy, allowing for flexible privacy guarantees.
* **Variety of Mechanisms:** Implements several common mechanisms, including:
    * **Laplace Mechanism:** For adding noise to real-valued queries with ε-differential privacy.
    * **Gaussian Mechanism:** For adding noise to real-valued queries, providing (ε, δ)-differential privacy.
    * **Exponential Mechanism:** For privately selecting an item from a set of candidates based on utility scores.
    * **Report Noisy Max:** For privately releasing the index of the maximum value in a set of counts.
* **Advanced Privacy Accounting:** Multiple composition methods for tighter privacy bounds:
    * **Basic Composition:** Simple additive composition (ε_total = Σε_i)
    * **Advanced Composition:** Tighter bounds using the Dwork-Rothblum-Vadhan theorem
    * **Optimal Composition:** Numerically optimized δ' allocation
    * **RDP Composition:** Rényi Differential Privacy for even tighter bounds
* **Budget Management:** Track and enforce privacy budgets across multiple queries.
* **Rust-based:** Benefits from Rust's performance, memory safety, and strong type system.
* **Comprehensive Testing:** Includes unit tests and statistical validation for all mechanisms.

## Getting Started

Add this library to your `Cargo.toml`:

```toml
[dependencies]
differential_privacy = "0.1.0"
```

## Usage Examples

### Laplace Mechanism

```rust
use differential_privacy::mechanisms::laplace_mechanism;
use differential_privacy::privacy_accounting::PrivacyAccountant;

fn main() {
    let value = 100.0;
    let sensitivity = 1.0;
    let epsilon = 0.5;
    let mut accountant = PrivacyAccountant::new();

    let noisy_value = laplace_mechanism(value, sensitivity, epsilon, &mut accountant);
    println!("Noisy Value: {}", noisy_value);

    let (total_epsilon, _) = accountant.compute_basic_composition();
    println!("Total Epsilon: {}", total_epsilon);
}
```

### Gaussian Mechanism

```rust
use differential_privacy::mechanisms::gaussian_mechanism;
use differential_privacy::privacy_accounting::PrivacyAccountant;

fn main() {
    let value = 100.0;
    let sensitivity = 1.0;
    let epsilon = 0.5;
    let delta = 1e-5;
    let mut accountant = PrivacyAccountant::new();

    let noisy_value = gaussian_mechanism(value, sensitivity, epsilon, delta, &mut accountant)
        .expect("Invalid parameters");
    println!("Noisy Value: {}", noisy_value);

    let (total_epsilon, total_delta) = accountant.compute_basic_composition();
    println!("Total Epsilon: {}, Total Delta: {}", total_epsilon, total_delta);
}
```

### Exponential Mechanism

```rust
use differential_privacy::mechanisms::exponential_mechanism;
use differential_privacy::privacy_accounting::PrivacyAccountant;

fn main() {
    // Select the best category privately based on utility scores
    let utilities = vec![10.0, 25.0, 15.0, 5.0];
    let sensitivity = 1.0;
    let epsilon = 0.5;
    let mut accountant = PrivacyAccountant::new();

    let selected_index = exponential_mechanism(&utilities, sensitivity, epsilon, &mut accountant)
        .expect("Invalid parameters");
    println!("Selected category index: {}", selected_index);
}
```

### Report Noisy Max

```rust
use differential_privacy::mechanisms::report_noisy_max;
use differential_privacy::privacy_accounting::PrivacyAccountant;

fn main() {
    // Find the winning option from vote counts
    let counts = vec![150.0, 200.0, 175.0, 50.0];
    let sensitivity = 1.0;
    let epsilon = 0.5;
    let mut accountant = PrivacyAccountant::new();

    let (winning_index, noisy_max) = report_noisy_max(&counts, sensitivity, epsilon, &mut accountant)
        .expect("Invalid parameters");
    println!("Winning option: {} (noisy count: {})", winning_index, noisy_max);
}
```

### Advanced Privacy Composition

The library supports multiple composition methods that provide tighter privacy bounds when running multiple queries:

```rust
use differential_privacy::privacy_accounting::{PrivacyAccountant, CompositionMethod};

fn main() {
    let mut accountant = PrivacyAccountant::new();

    // Run 100 queries with ε=0.1 each
    for _ in 0..100 {
        accountant.update(0.1, 0.0);
    }

    // Compare composition methods
    let (basic_eps, _) = accountant.get_privacy_loss(CompositionMethod::Basic);
    let (advanced_eps, _) = accountant.get_privacy_loss(CompositionMethod::Advanced);
    let (optimal_eps, _) = accountant.get_privacy_loss(CompositionMethod::OptimalAdvanced);

    println!("Basic composition:    ε = {:.2}", basic_eps);    // ε = 10.00
    println!("Advanced composition: ε = {:.2}", advanced_eps);  // ε ≈ 5.85
    println!("Optimal composition:  ε = {:.2}", optimal_eps);   // ε ≈ 5.50

    // Get a full comparison summary
    let summary = accountant.composition_summary(1e-5);
    println!("{}", summary);
}
```

### Budget Management

```rust
use differential_privacy::privacy_accounting::{PrivacyAccountant, CompositionMethod};

fn main() {
    // Create an accountant with a privacy budget
    let mut accountant = PrivacyAccountant::with_budget(1.0, 1e-5);

    // Run queries and check budget
    accountant.update(0.3, 1e-6);
    accountant.update(0.3, 1e-6);

    // Check remaining budget
    let (remaining_eps, remaining_delta) = accountant.get_remaining_budget(CompositionMethod::Basic);
    println!("Remaining: ε = {:?}, δ = {:?}", remaining_eps, remaining_delta);

    // Check if budget is exceeded
    if accountant.is_budget_exceeded(CompositionMethod::Basic) {
        println!("Warning: Privacy budget exceeded!");
    }
}
```

### Automatic Budget Enforcement

Use `try_update` to automatically check budget before recording queries:

```rust
use differential_privacy::privacy_accounting::PrivacyAccountant;

fn main() {
    let mut accountant = PrivacyAccountant::with_budget(1.0, 1e-5);

    // try_update checks budget before recording
    match accountant.try_update(0.5, 1e-6) {
        Ok(()) => println!("Query recorded successfully"),
        Err(e) => println!("Budget exceeded: {}", e),
    }

    // Check if we can afford a query before running
    if accountant.can_afford(0.3, 0.0).is_ok() {
        accountant.update(0.3, 0.0);
    }
}
```

### Budget-Enforcing Mechanisms

Use the `BudgetedAccountant` for automatic enforcement with all mechanisms:

```rust
use differential_privacy::mechanisms::BudgetedAccountant;

fn main() {
    let mut accountant = BudgetedAccountant::new(1.0, 1e-5);

    // All mechanism calls automatically check budget
    match accountant.laplace(100.0, 1.0, 0.3) {
        Ok(noisy_value) => println!("Result: {}", noisy_value),
        Err(e) => println!("Failed: {}", e),
    }

    // Check remaining budget
    let (remaining_eps, remaining_delta) = accountant.remaining_budget();
    println!("Remaining: ε = {:?}, δ = {:?}", remaining_eps, remaining_delta);

    // Queries that would exceed budget are rejected
    let result = accountant.laplace(100.0, 1.0, 0.8); // Would exceed budget
    assert!(result.is_err());
}
```

Or use individual budgeted mechanism functions:

```rust
use differential_privacy::mechanisms::laplace_mechanism_budgeted;
use differential_privacy::privacy_accounting::PrivacyAccountant;

fn main() {
    let mut accountant = PrivacyAccountant::with_budget(1.0, 1e-5);

    // Automatically checks and enforces budget
    match laplace_mechanism_budgeted(100.0, 1.0, 0.5, &mut accountant) {
        Ok(noisy_value) => println!("Result: {}", noisy_value),
        Err(e) => println!("Budget exceeded: {}", e),
    }
}
```

## API Reference

### Mechanisms

| Mechanism | Privacy Guarantee | Use Case |
|-----------|-------------------|----------|
| `laplace_mechanism` | ε-DP | Adding noise to numeric queries |
| `gaussian_mechanism` | (ε, δ)-DP | Adding noise when δ > 0 is acceptable |
| `exponential_mechanism` | ε-DP | Selecting from discrete candidates |
| `report_noisy_max` | ε-DP | Finding the argmax of counts |

### Composition Methods

| Method | Description | When to Use |
|--------|-------------|-------------|
| `Basic` | Simple sum: ε_total = Σε_i | Few queries, loose bounds acceptable |
| `Advanced` | Dwork-Rothblum-Vadhan theorem | Many queries with similar ε values |
| `OptimalAdvanced` | Numerically optimized | When tightest bounds needed |
| `RDP` | Rényi Differential Privacy | Gaussian mechanisms, ML training |

### Privacy Accountant Methods

```rust
// Create accountants
PrivacyAccountant::new()                          // No budget limits
PrivacyAccountant::with_budget(epsilon, delta)    // With budget limits

// Record queries
accountant.update(epsilon, delta)                 // Record without checking budget
accountant.try_update(epsilon, delta)             // Check budget, then record
accountant.try_update_with_method(eps, delta, method)  // With specific composition

// Check affordability
accountant.can_afford(epsilon, delta)             // Check if query fits in budget
accountant.can_afford_with_method(eps, delta, method)  // With specific composition

// Get privacy loss
accountant.compute_basic_composition()            // Basic composition
accountant.get_privacy_loss(method)               // Any composition method
accountant.get_privacy_loss_advanced(delta_prime) // Advanced with custom δ'
accountant.get_privacy_loss_optimal(target_delta) // Optimal composition
accountant.get_privacy_loss_rdp_optimal(delta)    // RDP composition

// Budget management
accountant.is_budget_exceeded(method)
accountant.get_remaining_budget(method)
accountant.has_budget()                           // Check if budget is set
accountant.get_budget()                           // Get budget limits
accountant.set_budget(Some(eps), Some(delta))     // Set new budget limits
accountant.num_queries()
accountant.reset()
```

### Budget-Enforcing Mechanisms

| Function | Description |
|----------|-------------|
| `laplace_mechanism_budgeted` | Laplace with automatic budget check |
| `gaussian_mechanism_budgeted` | Gaussian with automatic budget check |
| `exponential_mechanism_budgeted` | Exponential with automatic budget check |
| `report_noisy_max_budgeted` | Report Noisy Max with automatic budget check |
| `BudgetedAccountant` | Wrapper with all mechanisms built-in |

## Running Tests

```bash
cargo test
```

## Running Examples

```bash
cargo run --example noise_addition
```

## License

This project is licensed under the MIT License.
