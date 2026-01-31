# Differential Privacy Library in Rust

A robust, performant, and comprehensive toolkit for implementing differential privacy in data analysis pipelines. Built with Rust's safety guarantees and zero-cost abstractions.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Mechanisms](#core-mechanisms)
  - [Laplace Mechanism](#laplace-mechanism)
  - [Gaussian Mechanism](#gaussian-mechanism)
  - [Exponential Mechanism](#exponential-mechanism)
  - [Report Noisy Max](#report-noisy-max)
  - [Sparse Vector Technique](#sparse-vector-technique)
- [Privacy Amplification by Subsampling](#privacy-amplification-by-subsampling)
- [Private Aggregations](#private-aggregations)
- [Privacy Accounting](#privacy-accounting)
- [Budget Management](#budget-management)
- [API Reference](#api-reference)
- [Mathematical Background](#mathematical-background)
- [Testing](#testing)
- [License](#license)

## Overview

Differential privacy is a mathematical framework that provides strong privacy guarantees when analyzing sensitive data. This library implements the core building blocks for differentially private data analysis:

- **Noise Mechanisms**: Add calibrated random noise to query results
- **Selection Mechanisms**: Privately choose from discrete options
- **Threshold Mechanisms**: Answer many threshold queries efficiently
- **Privacy Accounting**: Track cumulative privacy loss across queries
- **Subsampling Amplification**: Strengthen privacy through random sampling

### What is Differential Privacy?

A randomized algorithm M satisfies (ε, δ)-differential privacy if for all datasets D and D' differing in one record, and all possible outputs S:

```
P[M(D) ∈ S] ≤ e^ε × P[M(D') ∈ S] + δ
```

- **ε (epsilon)**: Privacy budget. Smaller = stronger privacy.
- **δ (delta)**: Probability of privacy breach. Typically ≤ 1/n².

## Features

### Core Mechanisms
- **Laplace Mechanism**: ε-DP noise for numeric queries
- **Gaussian Mechanism**: (ε, δ)-DP noise with better composition
- **Exponential Mechanism**: Private selection from discrete candidates
- **Report Noisy Max**: Private argmax for counting queries
- **Sparse Vector Technique**: Answer many threshold queries with fixed budget

### Privacy Amplification
- **Poisson Subsampling**: Each record included with probability q
- **Uniform Subsampling**: Fixed-size random subset
- **Amplification Formulas**: Compute tighter privacy bounds

### Private Aggregations
- **Private Sum/Mean/Count**: Basic statistics with noise
- **Private Variance**: Bounded variance estimation
- **Private Histogram**: Per-bin noise for distributions
- **Vector Operations**: Noise on entire arrays

### Privacy Accounting
- **Basic Composition**: Simple additive bounds
- **Advanced Composition**: Tighter Dwork-Rothblum-Vadhan bounds
- **Optimal Composition**: Numerically optimized bounds
- **RDP Composition**: Rényi Differential Privacy

### Budget Management
- **Budget Enforcement**: Automatic limit checking
- **Detailed Errors**: Clear messages when budget exceeded
- **Flexible Limits**: Set ε and/or δ budgets

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
differential_privacy = "0.1.0"
ndarray = "0.15"  # Required for aggregation functions
```

## Quick Start

```rust
use differential_privacy::mechanisms::laplace_mechanism;
use differential_privacy::privacy_accounting::PrivacyAccountant;

fn main() {
    // Create a privacy accountant to track budget usage
    let mut accountant = PrivacyAccountant::new();

    // True query result (e.g., sum of salaries)
    let true_value = 1_000_000.0;

    // Add Laplace noise for privacy
    // sensitivity = max change from one person = 200000 (max salary)
    // epsilon = 1.0 (privacy budget)
    let noisy_value = laplace_mechanism(true_value, 200_000.0, 1.0, &mut accountant);

    println!("True value: ${:.2}", true_value);
    println!("Noisy value: ${:.2}", noisy_value);

    // Check privacy consumption
    let (total_eps, total_delta) = accountant.compute_basic_composition();
    println!("Privacy spent: ε = {}, δ = {}", total_eps, total_delta);
}
```

## Core Mechanisms

### Laplace Mechanism

The Laplace mechanism adds noise drawn from a Laplace distribution, providing pure ε-differential privacy. Best for numeric queries where you want simple privacy guarantees.

**When to use**: Counting, sums, averages with bounded sensitivity.

```rust
use differential_privacy::mechanisms::laplace_mechanism;
use differential_privacy::privacy_accounting::PrivacyAccountant;

fn main() {
    let mut accountant = PrivacyAccountant::new();

    // Example: Count of users over age 30
    let true_count = 1542.0;
    let sensitivity = 1.0;  // One person changes count by at most 1
    let epsilon = 0.5;      // Privacy budget

    let noisy_count = laplace_mechanism(true_count, sensitivity, epsilon, &mut accountant);

    println!("True count: {}", true_count);
    println!("Noisy count: {:.0}", noisy_count);

    // Expected noise magnitude: sensitivity / epsilon = 2.0
    // 95% of noise will be within ±6 of true value
}
```

**Privacy guarantee**: For sensitivity Δ and epsilon ε, noise is Laplace(Δ/ε).

### Gaussian Mechanism

The Gaussian mechanism adds normally-distributed noise, providing (ε, δ)-differential privacy. Better for composition (running many queries) due to tighter RDP bounds.

**When to use**: When δ > 0 is acceptable and you'll run many queries.

```rust
use differential_privacy::mechanisms::gaussian_mechanism;
use differential_privacy::privacy_accounting::PrivacyAccountant;

fn main() {
    let mut accountant = PrivacyAccountant::new();

    let true_value = 500.0;
    let sensitivity = 10.0;
    let epsilon = 1.0;
    let delta = 1e-5;  // Probability of privacy breach

    let noisy_value = gaussian_mechanism(true_value, sensitivity, epsilon, delta, &mut accountant)
        .expect("Invalid parameters");

    println!("True value: {}", true_value);
    println!("Noisy value: {:.2}", noisy_value);

    // Verify privacy tracking
    let (eps, del) = accountant.compute_basic_composition();
    println!("Privacy: ε = {}, δ = {}", eps, del);
}
```

**Privacy guarantee**: Noise is N(0, σ²) where σ = Δ × √(2 ln(1.25/δ)) / ε.

### Exponential Mechanism

The exponential mechanism privately selects an item from a discrete set based on utility scores. Items with higher utility are more likely to be selected.

**When to use**: Selecting categories, choosing features, picking locations.

```rust
use differential_privacy::mechanisms::exponential_mechanism;
use differential_privacy::privacy_accounting::PrivacyAccountant;

fn main() {
    let mut accountant = PrivacyAccountant::new();

    // Example: Choose the best marketing channel based on conversion rates
    let channels = vec!["Email", "Social", "Search", "Display"];
    let utilities = vec![15.2, 23.5, 18.1, 8.4];  // Conversion rates

    let sensitivity = 1.0;  // Max change in utility from one user
    let epsilon = 1.0;

    let selected_idx = exponential_mechanism(&utilities, sensitivity, epsilon, &mut accountant)
        .expect("Invalid parameters");

    println!("Selected channel: {} (utility: {:.1})",
             channels[selected_idx], utilities[selected_idx]);

    // Higher epsilon = more likely to pick the true best
    // Lower epsilon = more random selection for privacy
}
```

**Privacy guarantee**: Item i selected with probability ∝ exp(ε × utility[i] / (2Δ)).

### Report Noisy Max

Report Noisy Max finds the index of the maximum value in a set of counts, with privacy protection. Returns both the winning index and the noisy maximum value.

**When to use**: Finding the most popular category, winner of a vote, most common value.

```rust
use differential_privacy::mechanisms::report_noisy_max;
use differential_privacy::privacy_accounting::PrivacyAccountant;

fn main() {
    let mut accountant = PrivacyAccountant::new();

    // Example: Find the winning candidate in an election
    let candidates = vec!["Alice", "Bob", "Carol", "David"];
    let vote_counts = vec![1523.0, 1891.0, 1456.0, 892.0];

    let sensitivity = 1.0;  // One person affects one count by 1
    let epsilon = 0.5;

    let (winner_idx, noisy_count) = report_noisy_max(
        &vote_counts, sensitivity, epsilon, &mut accountant
    ).expect("Invalid parameters");

    println!("Winner: {} with approximately {:.0} votes",
             candidates[winner_idx], noisy_count);
}
```

### Sparse Vector Technique

The Sparse Vector Technique (SVT) answers many threshold queries ("Is query result ≥ T?") with a fixed privacy budget. Only queries that exceed the threshold consume significant privacy.

**When to use**: Finding rare events, anomaly detection, identifying values above a threshold.

```rust
use differential_privacy::mechanisms::{SparseVectorTechnique, ThresholdResult};
use differential_privacy::privacy_accounting::PrivacyAccountant;

fn main() {
    let mut accountant = PrivacyAccountant::new();

    // Find days with unusually high traffic (> 10000 visitors)
    let daily_traffic = vec![
        8500.0, 9200.0, 12500.0, 7800.0, 15200.0,
        9100.0, 8900.0, 11000.0, 6500.0, 18000.0
    ];

    // Create SVT: finds up to 3 days above threshold 10000
    let mut svt = SparseVectorTechnique::new(
        10000.0,  // threshold
        1.0,      // sensitivity (one user = one visit)
        1.0,      // total epsilon budget
        3,        // max "above threshold" answers
        &mut accountant
    ).expect("Invalid parameters");

    println!("Days with high traffic:");
    for (day, &traffic) in daily_traffic.iter().enumerate() {
        match svt.query(traffic) {
            Some(ThresholdResult::Above) => {
                println!("  Day {}: HIGH (actual: {:.0})", day + 1, traffic);
            }
            Some(ThresholdResult::Below) => {
                // Don't reveal anything about below-threshold days
            }
            None => {
                println!("  (SVT exhausted - found {} high-traffic days)", svt.above_count());
                break;
            }
        }
    }

    // Key benefit: Fixed privacy cost regardless of how many days we check!
    let (eps, _) = accountant.compute_basic_composition();
    println!("\nTotal privacy cost: ε = {} (fixed!)", eps);
}
```

**Convenience functions**:

```rust
use differential_privacy::mechanisms::{sparse_vector_find_first, sparse_vector_find_all};
use differential_privacy::privacy_accounting::PrivacyAccountant;

fn main() {
    let mut accountant = PrivacyAccountant::new();
    let values = vec![50.0, 80.0, 120.0, 90.0, 150.0, 200.0, 75.0];

    // Find first value above 100
    if let Some(idx) = sparse_vector_find_first(&values, 100.0, 1.0, 0.5, &mut accountant)
        .expect("Invalid parameters")
    {
        println!("First value above 100 at index {}", idx);
    }

    // Find all values above 100 (up to 3)
    let mut accountant2 = PrivacyAccountant::new();
    let indices = sparse_vector_find_all(&values, 100.0, 1.0, 0.5, 3, &mut accountant2)
        .expect("Invalid parameters");
    println!("Indices above 100: {:?}", indices);
}
```

## Privacy Amplification by Subsampling

When you apply a DP mechanism to a random subsample of your data, privacy guarantees are amplified. This is crucial for:

- **DP-SGD**: Mini-batch training in machine learning
- **Large-scale analytics**: Analyzing random samples
- **Streaming**: Processing subsets of data streams

### Key Insight

If you sample each record with probability q = 1%, a mechanism with ε = 1.0 achieves effective privacy of ε ≈ 0.01!

### Computing Amplified Privacy

```rust
use differential_privacy::mechanisms::{
    amplify_epsilon_poisson,
    amplify_epsilon_uniform,
    amplify_epsilon_delta_poisson,
    compute_base_epsilon,
    SubsampledMechanism,
};

fn main() {
    // Poisson subsampling: each record included with probability q
    let base_epsilon = 1.0;
    let sampling_rate = 0.01;  // 1%

    let amplified = amplify_epsilon_poisson(base_epsilon, sampling_rate)
        .expect("Valid parameters");
    println!("Base ε: {}, Amplified ε: {:.4}", base_epsilon, amplified);
    // Output: Base ε: 1, Amplified ε: 0.0101

    // Uniform subsampling: sample k records from n total
    let amplified_uniform = amplify_epsilon_uniform(1.0, 100, 10000)
        .expect("Valid parameters");
    println!("Uniform (100/10000): Amplified ε: {:.4}", amplified_uniform);

    // For (ε, δ)-DP mechanisms
    let (amp_eps, amp_delta) = amplify_epsilon_delta_poisson(1.0, 1e-5, 0.01)
        .expect("Valid parameters");
    println!("Amplified: ε = {:.4}, δ = {:.2e}", amp_eps, amp_delta);

    // Compute base epsilon needed for target amplified epsilon
    let target_eps = 0.1;
    let base_needed = compute_base_epsilon(target_eps, 0.01)
        .expect("Valid parameters");
    println!("To achieve ε = {} with q = 1%, need base ε = {:.2}", target_eps, base_needed);

    // Use SubsampledMechanism for convenient configuration
    let mech = SubsampledMechanism::new_poisson(2.0, 1e-5, 0.01)
        .expect("Valid parameters");
    println!("Amplification factor: {:.1}x", mech.amplification_factor());
}
```

### Applying Subsampled Mechanisms

```rust
use differential_privacy::mechanisms::{
    subsampled_laplace_sum,
    subsampled_laplace_mean,
    uniform_subsample,
    poisson_subsample,
};
use differential_privacy::privacy_accounting::PrivacyAccountant;
use ndarray::Array1;

fn main() {
    let mut accountant = PrivacyAccountant::new();

    // Create a large dataset
    let data: Array1<f64> = Array1::from_vec(
        (0..100_000).map(|i| (i % 1000) as f64).collect()
    );

    // Subsampled sum: automatically subsamples and adds noise
    let noisy_sum = subsampled_laplace_sum(
        data.view(),
        1000.0,  // sensitivity (max value)
        2.0,     // base epsilon (can be larger due to amplification)
        0.01,    // sampling probability (1%)
        &mut accountant
    ).expect("Valid parameters");

    let (recorded_eps, _) = accountant.compute_basic_composition();
    println!("Noisy sum: {:.2}", noisy_sum);
    println!("Privacy cost (amplified): {:.4}", recorded_eps);
    // recorded_eps is much smaller than 2.0!

    // Subsampled mean
    let mut accountant2 = PrivacyAccountant::new();
    let noisy_mean = subsampled_laplace_mean(
        data.view(),
        0.0,     // lower bound
        1000.0,  // upper bound
        1.0,     // base epsilon
        0.1,     // 10% sampling
        &mut accountant2
    ).expect("Valid parameters");
    println!("Noisy mean: {:.2}", noisy_mean);

    // Manual subsampling for custom mechanisms
    let (sample, indices) = uniform_subsample(data.view(), 1000)
        .expect("Valid size");
    println!("Sampled {} elements", sample.len());

    // Or Poisson subsampling
    let poisson_sample = poisson_subsample(data.view(), 0.01);
    println!("Poisson sample size: {} (expected ~1000)", poisson_sample.len());
}
```

## Private Aggregations

The library provides ready-to-use differentially private aggregation functions.

### Private Sum and Mean

```rust
use differential_privacy::aggregations::{private_sum, private_mean};
use differential_privacy::privacy_accounting::PrivacyAccountant;
use ndarray::array;

fn main() {
    let mut accountant = PrivacyAccountant::new();

    // Salaries (bounded between 30k and 200k)
    let salaries = array![75000.0, 120000.0, 85000.0, 95000.0, 150000.0];

    // Private sum
    let noisy_sum = private_sum(
        salaries.view(),
        30000.0,   // lower bound
        200000.0,  // upper bound
        0.5,       // epsilon
        &mut accountant
    ).expect("Invalid parameters");

    println!("Noisy total salaries: ${:.2}", noisy_sum);

    // Private mean
    let noisy_mean = private_mean(
        salaries.view(),
        30000.0,
        200000.0,
        0.5,
        &mut accountant
    ).expect("Invalid parameters");

    println!("Noisy average salary: ${:.2}", noisy_mean);
}
```

### Private Count

```rust
use differential_privacy::aggregations::{private_count, private_count_all};
use differential_privacy::privacy_accounting::PrivacyAccountant;
use ndarray::array;

fn main() {
    let mut accountant = PrivacyAccountant::new();

    let ages = array![25.0, 35.0, 42.0, 28.0, 55.0, 31.0, 47.0, 23.0];

    // Count people over 30
    let count_over_30 = private_count(
        ages.view(),
        |&age| age > 30.0,  // predicate
        0.5,                // epsilon
        &mut accountant
    );

    println!("People over 30: {:.1}", count_over_30);

    // Total count
    let total = private_count_all(ages.view(), 0.5, &mut accountant);
    println!("Total people: {:.1}", total);
}
```

### Private Histogram

```rust
use differential_privacy::aggregations::private_histogram;
use differential_privacy::privacy_accounting::PrivacyAccountant;
use ndarray::array;

fn main() {
    let mut accountant = PrivacyAccountant::new();

    let ages = array![22.0, 35.0, 45.0, 28.0, 52.0, 38.0, 25.0, 60.0, 33.0, 41.0];

    // Age groups: 0-29, 30-39, 40-49, 50+
    let bins = vec![0.0, 30.0, 40.0, 50.0, 100.0];

    let histogram = private_histogram(ages.view(), &bins, 0.5, &mut accountant)
        .expect("Invalid parameters");

    println!("Age Distribution:");
    println!("  18-29: {:.1}", histogram[0]);
    println!("  30-39: {:.1}", histogram[1]);
    println!("  40-49: {:.1}", histogram[2]);
    println!("  50+:   {:.1}", histogram[3]);
}
```

### Vector Noise

```rust
use differential_privacy::aggregations::{add_laplace_noise_vector, add_gaussian_noise_vector};
use differential_privacy::privacy_accounting::PrivacyAccountant;
use ndarray::array;

fn main() {
    let mut accountant = PrivacyAccountant::new();

    let features = array![100.0, 200.0, 150.0, 175.0, 125.0];

    // Add Laplace noise to entire vector
    let noisy_laplace = add_laplace_noise_vector(
        features.view(),
        10.0,  // sensitivity per element
        1.0,   // epsilon
        &mut accountant
    );

    println!("Original: {:?}", features);
    println!("Noisy (Laplace): {:?}", noisy_laplace);

    // Add Gaussian noise
    let noisy_gaussian = add_gaussian_noise_vector(
        features.view(),
        10.0,   // sensitivity
        1.0,    // epsilon
        1e-5,   // delta
        &mut accountant
    ).expect("Invalid parameters");

    println!("Noisy (Gaussian): {:?}", noisy_gaussian);
}
```

## Privacy Accounting

Track cumulative privacy loss across multiple queries using different composition methods.

### Composition Methods

```rust
use differential_privacy::privacy_accounting::{PrivacyAccountant, CompositionMethod};

fn main() {
    let mut accountant = PrivacyAccountant::new();

    // Simulate 100 queries with ε=0.1 each
    for _ in 0..100 {
        accountant.update(0.1, 0.0);
    }

    // Compare composition methods
    let (basic_eps, _) = accountant.get_privacy_loss(CompositionMethod::Basic);
    let (advanced_eps, _) = accountant.get_privacy_loss(CompositionMethod::Advanced);
    let (optimal_eps, _) = accountant.get_privacy_loss(CompositionMethod::OptimalAdvanced);

    println!("100 queries with ε=0.1 each:");
    println!("  Basic composition:    ε = {:.2}", basic_eps);     // 10.00
    println!("  Advanced composition: ε = {:.2}", advanced_eps);  // ~5.85
    println!("  Optimal composition:  ε = {:.2}", optimal_eps);   // ~5.50

    // Full summary
    let summary = accountant.composition_summary(1e-5);
    println!("\n{}", summary);
}
```

### RDP Composition

Rényi Differential Privacy provides even tighter bounds for Gaussian mechanisms:

```rust
use differential_privacy::privacy_accounting::PrivacyAccountant;

fn main() {
    let mut accountant = PrivacyAccountant::new();

    // Simulate 50 Gaussian mechanism queries
    for _ in 0..50 {
        accountant.update(0.5, 1e-6);
    }

    // RDP composition with optimal order selection
    let (rdp_eps, rdp_delta) = accountant.get_privacy_loss_rdp_optimal(1e-5);
    let (basic_eps, _) = accountant.compute_basic_composition();

    println!("50 Gaussian queries:");
    println!("  Basic:  ε = {:.2}", basic_eps);
    println!("  RDP:    ε = {:.2} (at δ = {:.0e})", rdp_eps, rdp_delta);
}
```

## Budget Management

Set and enforce privacy budgets to prevent overspending.

### Setting Budgets

```rust
use differential_privacy::privacy_accounting::PrivacyAccountant;

fn main() {
    // Create accountant with budget: ε ≤ 1.0, δ ≤ 10⁻⁵
    let mut accountant = PrivacyAccountant::with_budget(1.0, 1e-5);

    // Record queries
    accountant.update(0.3, 1e-6);
    accountant.update(0.3, 1e-6);

    // Check status
    println!("Budget: ε = 1.0, δ = 1e-5");
    println!("Spent:  ε = 0.6, δ = 2e-6");

    let (remaining_eps, remaining_delta) = accountant.get_remaining_budget(
        differential_privacy::privacy_accounting::CompositionMethod::Basic
    );
    println!("Remaining: ε = {:?}, δ = {:?}", remaining_eps, remaining_delta);
}
```

### Automatic Budget Enforcement

```rust
use differential_privacy::privacy_accounting::{PrivacyAccountant, PrivacyBudgetError};

fn main() {
    let mut accountant = PrivacyAccountant::with_budget(1.0, 1e-5);

    // try_update checks budget before recording
    match accountant.try_update(0.6, 1e-6) {
        Ok(()) => println!("Query 1 recorded"),
        Err(e) => println!("Query 1 failed: {}", e),
    }

    match accountant.try_update(0.6, 1e-6) {
        Ok(()) => println!("Query 2 recorded"),
        Err(PrivacyBudgetError::EpsilonBudgetExceeded { requested, available, .. }) => {
            println!("Query 2 rejected: requested ε={:.2}, only {:.2} available",
                     requested, available);
        }
        Err(e) => println!("Query 2 failed: {}", e),
    }

    // Check affordability before running expensive computation
    if accountant.can_afford(0.2, 0.0).is_ok() {
        println!("Can afford one more query with ε=0.2");
    }
}
```

### Budget-Enforcing Mechanisms

Use the `BudgetedAccountant` wrapper for automatic enforcement:

```rust
use differential_privacy::mechanisms::BudgetedAccountant;

fn main() {
    let mut accountant = BudgetedAccountant::new(1.0, 1e-5);

    // All mechanism calls automatically check budget
    match accountant.laplace(100.0, 1.0, 0.3) {
        Ok(noisy_value) => println!("Query 1: {:.2}", noisy_value),
        Err(e) => println!("Query 1 failed: {}", e),
    }

    match accountant.gaussian(100.0, 1.0, 0.5, 1e-6) {
        Ok(noisy_value) => println!("Query 2: {:.2}", noisy_value),
        Err(e) => println!("Query 2 failed: {}", e),
    }

    // Check remaining budget
    let (remaining_eps, remaining_delta) = accountant.remaining_budget();
    println!("Remaining: ε = {:?}, δ = {:?}", remaining_eps, remaining_delta);

    // This will fail if it exceeds budget
    match accountant.laplace(100.0, 1.0, 0.5) {
        Ok(_) => println!("Query 3 succeeded"),
        Err(e) => println!("Query 3 rejected: {}", e),
    }
}
```

## API Reference

### Mechanisms

| Function | Privacy | Description |
|----------|---------|-------------|
| `laplace_mechanism` | ε-DP | Laplace noise for numeric queries |
| `gaussian_mechanism` | (ε,δ)-DP | Gaussian noise for numeric queries |
| `exponential_mechanism` | ε-DP | Private selection from candidates |
| `report_noisy_max` | ε-DP | Private argmax with noisy count |
| `report_noisy_argmax` | ε-DP | Private argmax (index only) |

### Sparse Vector Technique

| Type/Function | Description |
|---------------|-------------|
| `SparseVectorTechnique` | Streaming threshold queries (Above/Below) |
| `NumericSparseVector` | Threshold queries with noisy output values |
| `sparse_vector_find_first` | Find first value above threshold |
| `sparse_vector_find_all` | Find all values above threshold (up to max) |

### Privacy Amplification

| Function | Description |
|----------|-------------|
| `amplify_epsilon_poisson` | Amplified ε for Poisson sampling |
| `amplify_epsilon_uniform` | Amplified ε for uniform sampling |
| `amplify_epsilon_delta_poisson` | Amplified (ε,δ) for approximate DP |
| `compute_base_epsilon` | Base ε needed for target amplified ε |
| `SubsampledMechanism` | Configuration helper |
| `poisson_subsample` | Sample with probability q |
| `uniform_subsample` | Sample k from n without replacement |
| `subsampled_laplace_sum` | Subsampled sum with noise |
| `subsampled_laplace_mean` | Subsampled mean with noise |

### Aggregations

| Function | Description |
|----------|-------------|
| `private_sum` | Sum with Laplace noise |
| `private_mean` | Mean with noise on sum and count |
| `private_count` | Count matching predicate |
| `private_count_all` | Total count |
| `private_variance` | Variance with bounded sensitivity |
| `private_histogram` | Histogram with per-bin noise |
| `add_laplace_noise_vector` | Laplace noise on arrays |
| `add_gaussian_noise_vector` | Gaussian noise on arrays |

### Sensitivity Calculators

| Function | Description |
|----------|-------------|
| `sensitivity_count` | Always 1.0 |
| `sensitivity_sum(lo, hi)` | hi - lo |
| `sensitivity_mean(lo, hi, n)` | (hi - lo) / n |
| `sensitivity_l2(dim)` | √dim for unit vectors |

### Privacy Accountant

```rust
// Creation
PrivacyAccountant::new()                          // No budget
PrivacyAccountant::with_budget(epsilon, delta)    // With budget

// Recording queries
accountant.update(epsilon, delta)                 // Record without checking
accountant.try_update(epsilon, delta)             // Check budget first
accountant.try_update_with_method(eps, delta, method)

// Budget checking
accountant.can_afford(epsilon, delta)
accountant.can_afford_with_method(eps, delta, method)

// Privacy loss
accountant.compute_basic_composition()
accountant.get_privacy_loss(method)
accountant.get_privacy_loss_advanced(delta_prime)
accountant.get_privacy_loss_optimal(target_delta)
accountant.get_privacy_loss_rdp_optimal(delta)

// Budget management
accountant.has_budget()
accountant.get_budget()
accountant.set_budget(Some(eps), Some(delta))
accountant.is_budget_exceeded(method)
accountant.get_remaining_budget(method)
accountant.num_queries()
accountant.reset()
```

## Mathematical Background

### Laplace Mechanism
- **Noise**: Laplace(0, Δ/ε)
- **Variance**: 2(Δ/ε)²
- **Privacy**: ε-differential privacy

### Gaussian Mechanism
- **Noise**: N(0, σ²) where σ = Δ√(2ln(1.25/δ))/ε
- **Privacy**: (ε, δ)-differential privacy

### Subsampling Amplification
- **Formula**: ε' = ln(1 + q(e^ε - 1))
- **Approximation**: ε' ≈ qε for small ε
- **Intuition**: If data is sampled with prob q, adversary's information is reduced

### Composition
- **Basic**: ε_total = Σε_i (loose)
- **Advanced**: ε_total = √(2k·ln(1/δ'))·ε + k·ε·(e^ε - 1) (tighter)
- **RDP**: Uses Rényi divergence for Gaussian mechanisms

## Testing

Run the test suite:

```bash
# All tests
cargo test

# With output
cargo test -- --nocapture

# Specific module
cargo test mechanisms::laplace

# Documentation tests
cargo test --doc
```

Current test coverage: 123 unit tests + 26 doc-tests.

## Examples

Run the included examples:

```bash
cargo run --example noise_addition
```

## License

This project is licensed under the MIT License.

## References

- Dwork, C., & Roth, A. (2014). The Algorithmic Foundations of Differential Privacy.
- Mironov, I. (2017). Rényi Differential Privacy.
- Abadi, M., et al. (2016). Deep Learning with Differential Privacy.
- Balle, B., et al. (2018). Privacy Amplification by Subsampling.
