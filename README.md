# Differential Privacy Library in Rust

A robust, performant, and easy-to-use toolkit for incorporating privacy-preserving techniques into data analysis pipelines.

## Features

* **Epsilon-Delta Differential Privacy:** Supports both ε and (ε, δ)-differential privacy, allowing for flexible privacy guarantees.
* **Variety of Mechanisms:** Implements several common mechanisms, including:
    * **Laplace Mechanism:** For adding noise to real-valued queries with ε-differential privacy.
    * **Gaussian Mechanism:** For adding noise to real-valued queries, providing (ε, δ)-differential privacy.
    * **Exponential Mechanism:** For privately selecting an item from a set of candidates based on utility scores.
    * **Report Noisy Max:** For privately releasing the index of the maximum value in a set of counts.
* **Privacy Accounting:** Built-in privacy accountant to track cumulative privacy loss across multiple queries.
* **Composable:** Mechanisms can be chained together while maintaining accurate privacy accounting.
* **Rust-based:** Benefits from Rust's performance, memory safety, and strong type system.
* **Clear API:** Designed for ease of use and integration into existing Rust projects.
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
    let mut accountant = PrivacyAccountant::new(0.0, 0.0);

    let noisy_value = laplace_mechanism(value, sensitivity, epsilon, &mut accountant);
    println!("Noisy Value: {}", noisy_value);

    let (total_epsilon, _) = accountant.get_privacy_loss();
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
    let mut accountant = PrivacyAccountant::new(0.0, 0.0);

    let noisy_value = gaussian_mechanism(value, sensitivity, epsilon, delta, &mut accountant)
        .expect("Invalid parameters");
    println!("Noisy Value: {}", noisy_value);

    let (total_epsilon, total_delta) = accountant.get_privacy_loss();
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
    let mut accountant = PrivacyAccountant::new(0.0, 0.0);

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
    let mut accountant = PrivacyAccountant::new(0.0, 0.0);

    let (winning_index, noisy_max) = report_noisy_max(&counts, sensitivity, epsilon, &mut accountant)
        .expect("Invalid parameters");
    println!("Winning option: {} (noisy count: {})", winning_index, noisy_max);
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

### Privacy Accounting

The `PrivacyAccountant` struct tracks cumulative privacy loss:

```rust
use differential_privacy::privacy_accounting::PrivacyAccountant;

let mut accountant = PrivacyAccountant::new(0.0, 0.0);

// After multiple mechanism calls...
let (total_epsilon, total_delta) = accountant.get_privacy_loss();
```

## Running Tests

```bash
cargo test
```

## License

This project is licensed under the MIT License.
