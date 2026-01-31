# Differential Privacy Library in Rust

A robust, performant, and easy-to-use toolkit for incorporating privacy-preserving techniques into data analysis pipelines.

## Features

* **Epsilon-Delta Differential Privacy:** Supports both ε and (ε, δ)-differential privacy, allowing for flexible privacy guarantees.
* **Variety of Mechanisms:** Implements several common mechanisms, including:
    * **Laplace Mechanism:** For adding noise to real-valued queries with ε-differential privacy.
    * **Gaussian Mechanism:** For adding noise to real-valued queries, providing (ε, δ)-differential privacy.
    * **Exponential Mechanism:** For privately selecting an item from a set of candidates based on utility scores.
    * **Report Noisy Max:** For privately releasing the index of the maximum value in a set of counts.
    * **Sparse Vector Technique:** Answer many threshold queries with a fixed privacy budget.
* **Privacy Amplification by Subsampling:** Stronger guarantees when using data subsamples:
    * **Poisson Subsampling:** Each record included independently with probability q
    * **Uniform Subsampling:** Fixed-size random subset without replacement
    * **Amplified Privacy:** Effective ε ≈ q × base_ε for small sampling rates
* **Advanced Privacy Accounting:** Multiple composition methods for tighter privacy bounds:
    * **Basic Composition:** Simple additive composition (ε_total = Σε_i)
    * **Advanced Composition:** Tighter bounds using the Dwork-Rothblum-Vadhan theorem
    * **Optimal Composition:** Numerically optimized δ' allocation
    * **RDP Composition:** Rényi Differential Privacy for even tighter bounds
* **Budget Management:** Track and enforce privacy budgets across multiple queries.
* **Private Aggregations:** Built-in differentially private aggregation functions:
    * **Private Sum/Mean/Count:** Compute statistics with Laplace noise
    * **Private Variance:** Bounded variance estimation
    * **Private Histogram:** Per-bin noise for histogram queries
    * **Vector Noise:** Add Laplace/Gaussian noise to entire arrays
    * **Sensitivity Calculators:** Utilities for computing query sensitivity
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

### Sparse Vector Technique

The Sparse Vector Technique (SVT) answers many threshold queries with a fixed privacy budget. It's ideal when you expect most queries to be below a threshold and want to identify the few that exceed it.

```rust
use differential_privacy::mechanisms::{SparseVectorTechnique, ThresholdResult};
use differential_privacy::privacy_accounting::PrivacyAccountant;

fn main() {
    let mut accountant = PrivacyAccountant::new();

    // Create SVT: threshold=100, sensitivity=1, epsilon=1.0, max 3 "Above" answers
    let mut svt = SparseVectorTechnique::new(100.0, 1.0, 1.0, 3, &mut accountant)
        .expect("Invalid parameters");

    // Process queries one at a time
    let queries = vec![50.0, 120.0, 80.0, 150.0, 90.0, 200.0];

    for query_value in queries {
        match svt.query(query_value) {
            Some(ThresholdResult::Above) => println!("{}: Above threshold!", query_value),
            Some(ThresholdResult::Below) => println!("{}: Below threshold", query_value),
            None => {
                println!("SVT exhausted (found max above-threshold answers)");
                break;
            }
        }
    }

    println!("Found {} queries above threshold", svt.above_count());
}
```

For convenience, use the helper functions:

```rust
use differential_privacy::mechanisms::{sparse_vector_find_first, sparse_vector_find_all};
use differential_privacy::privacy_accounting::PrivacyAccountant;

fn main() {
    let mut accountant = PrivacyAccountant::new();
    let queries = vec![50.0, 80.0, 120.0, 90.0, 150.0, 200.0];

    // Find the first query above threshold
    if let Some(index) = sparse_vector_find_first(&queries, 100.0, 1.0, 0.5, &mut accountant)
        .expect("Invalid parameters")
    {
        println!("First above threshold at index: {}", index);
    }

    // Find all queries above threshold (up to max 3)
    let mut accountant2 = PrivacyAccountant::new();
    let above_indices = sparse_vector_find_all(&queries, 100.0, 1.0, 0.5, 3, &mut accountant2)
        .expect("Invalid parameters");
    println!("Queries above threshold at indices: {:?}", above_indices);
}
```

### Privacy Amplification by Subsampling

When a DP mechanism is applied to a random subsample of data, privacy guarantees are amplified. This is crucial for DP-SGD in machine learning and large-scale analytics.

```rust
use differential_privacy::mechanisms::{
    amplify_epsilon_poisson, compute_base_epsilon, SubsampledMechanism
};

fn main() {
    // Compute amplified privacy for Poisson subsampling
    // Base mechanism: ε=1.0, sampling probability: 1%
    let amplified_eps = amplify_epsilon_poisson(1.0, 0.01)
        .expect("Valid parameters");
    println!("Base ε: 1.0, Amplified ε: {:.4}", amplified_eps);  // ~0.01

    // Compute what base epsilon is needed for a target amplified epsilon
    let base_eps = compute_base_epsilon(0.1, 0.01).expect("Valid parameters");
    println!("To achieve ε=0.1 with q=0.01, need base ε: {:.2}", base_eps);

    // Use SubsampledMechanism for convenient configuration
    let mech = SubsampledMechanism::new_poisson(2.0, 1e-5, 0.01)
        .expect("Valid parameters");
    println!("Amplification factor: {:.1}x", mech.amplification_factor());
}
```

Apply subsampled mechanisms to data:

```rust
use differential_privacy::mechanisms::{subsampled_laplace_sum, uniform_subsample};
use differential_privacy::privacy_accounting::PrivacyAccountant;
use ndarray::Array1;

fn main() {
    let mut accountant = PrivacyAccountant::new();

    // Create a dataset
    let data = Array1::from_vec((0..10000).map(|x| x as f64 % 100.0).collect());

    // Compute a subsampled sum with privacy amplification
    let noisy_sum = subsampled_laplace_sum(
        data.view(),
        100.0,  // sensitivity
        2.0,    // base epsilon (larger = less noise)
        0.01,   // sampling probability (1%)
        &mut accountant
    ).expect("Valid parameters");

    let (recorded_eps, _) = accountant.compute_basic_composition();
    println!("Noisy sum: {:.2}", noisy_sum);
    println!("Privacy cost (amplified): {:.4}", recorded_eps);  // Much less than 2.0!

    // Or manually subsample and apply your own mechanism
    let (sample, indices) = uniform_subsample(data.view(), 100).expect("Valid size");
    println!("Sampled {} elements at indices: {:?}...", sample.len(), &indices[..5]);
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

### Private Aggregations

The library provides differentially private versions of common statistical aggregations:

```rust
use differential_privacy::aggregations::{private_sum, private_mean, private_count};
use differential_privacy::privacy_accounting::PrivacyAccountant;
use ndarray::array;

fn main() {
    let mut accountant = PrivacyAccountant::new();
    let data = array![10.0, 20.0, 30.0, 40.0, 50.0];

    // Private sum with bounded values
    let noisy_sum = private_sum(data.view(), 0.0, 100.0, 0.5, &mut accountant)
        .expect("Invalid parameters");
    println!("Private sum: {}", noisy_sum);

    // Private mean
    let noisy_mean = private_mean(data.view(), 0.0, 100.0, 0.5, &mut accountant)
        .expect("Invalid parameters");
    println!("Private mean: {}", noisy_mean);

    // Private count with predicate
    let count = private_count(data.view(), |&x| x > 25.0, 0.5, &mut accountant);
    println!("Private count (x > 25): {}", count);
}
```

### Private Histogram

```rust
use differential_privacy::aggregations::private_histogram;
use differential_privacy::privacy_accounting::PrivacyAccountant;
use ndarray::array;

fn main() {
    let mut accountant = PrivacyAccountant::new();
    let ages = array![22.0, 35.0, 45.0, 28.0, 52.0, 38.0, 25.0, 60.0];
    let bins = vec![0.0, 30.0, 40.0, 50.0, 100.0];  // Age groups

    let noisy_histogram = private_histogram(ages.view(), &bins, 0.5, &mut accountant)
        .expect("Invalid parameters");

    println!("Age distribution:");
    println!("  0-29:   {:.1}", noisy_histogram[0]);
    println!("  30-39:  {:.1}", noisy_histogram[1]);
    println!("  40-49:  {:.1}", noisy_histogram[2]);
    println!("  50-99:  {:.1}", noisy_histogram[3]);
}
```

### Vector Noise Operations

```rust
use differential_privacy::aggregations::{add_laplace_noise_vector, add_gaussian_noise_vector};
use differential_privacy::privacy_accounting::PrivacyAccountant;
use ndarray::array;

fn main() {
    let mut accountant = PrivacyAccountant::new();
    let values = array![100.0, 200.0, 300.0];

    // Add Laplace noise to entire vector
    let noisy = add_laplace_noise_vector(values.view(), 1.0, 0.5, &mut accountant);
    println!("Noisy vector (Laplace): {:?}", noisy);

    // Add Gaussian noise to entire vector
    let noisy = add_gaussian_noise_vector(values.view(), 1.0, 0.5, 1e-5, &mut accountant)
        .expect("Invalid parameters");
    println!("Noisy vector (Gaussian): {:?}", noisy);
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
| `SparseVectorTechnique` | ε-DP | Many threshold queries with fixed budget |
| `NumericSparseVector` | ε-DP | SVT with noisy output values |

### Sparse Vector Technique

| Function/Type | Description |
|---------------|-------------|
| `SparseVectorTechnique` | Basic SVT returning Above/Below for each query |
| `NumericSparseVector` | SVT that also outputs noisy values for above-threshold |
| `sparse_vector_find_first` | Find first query exceeding threshold |
| `sparse_vector_find_all` | Find all queries exceeding threshold (up to max) |
| `ThresholdResult` | Enum: `Above` or `Below` |
| `NumericThresholdResult` | Enum: `Above(f64)` or `Below` |

### Privacy Amplification by Subsampling

| Function/Type | Description |
|---------------|-------------|
| `amplify_epsilon_poisson` | Compute amplified ε for Poisson subsampling |
| `amplify_epsilon_uniform` | Compute amplified ε for uniform subsampling |
| `amplify_epsilon_delta_poisson` | Compute amplified (ε, δ) for approximate DP |
| `compute_base_epsilon` | Compute base ε needed for target amplified ε |
| `poisson_subsample` | Sample data with Poisson subsampling |
| `poisson_subsample_indices` | Get indices of Poisson-sampled elements |
| `uniform_subsample` | Sample fixed-size subset without replacement |
| `SubsampledMechanism` | Configuration for subsampled mechanism |
| `subsampled_laplace_sum` | Laplace sum with privacy amplification |
| `subsampled_laplace_mean` | Laplace mean with privacy amplification |
| `subsampling_noise_reduction` | Estimate noise reduction factor |

### Aggregation Functions

| Function | Description |
|----------|-------------|
| `private_sum` | Sum with Laplace noise, bounded input |
| `private_mean` | Mean with noise on sum and count |
| `private_count` | Count with Laplace noise |
| `private_count_all` | Total count (no predicate) |
| `private_variance` | Variance with bounded sensitivity |
| `private_histogram` | Per-bin Laplace noise |
| `add_laplace_noise_vector` | Laplace noise on arrays |
| `add_gaussian_noise_vector` | Gaussian noise on arrays |

### Sensitivity Calculators

| Function | Description |
|----------|-------------|
| `sensitivity_count` | Count query sensitivity (always 1.0) |
| `sensitivity_sum(lower, upper)` | Sum sensitivity for bounded values |
| `sensitivity_mean(lower, upper, n)` | Mean sensitivity |
| `sensitivity_l2(dimension)` | L2 sensitivity for unit vectors |

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
