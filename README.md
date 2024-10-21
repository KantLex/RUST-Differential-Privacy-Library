WIP

# Differential Privacy Library in Rust

This library provides a collection of differentially private algorithms implemented in Rust.  It aims to offer a robust, performant, and easy-to-use toolkit for developers seeking to incorporate privacy-preserving techniques into their data analysis pipelines.

## Features

* **Epsilon-Delta Differential Privacy:**  Supports both ε-δ differential privacy, allowing for flexible privacy guarantees.
* **Variety of Mechanisms:** Implements several common mechanisms, including:
    * **Laplace Mechanism:**  For adding noise to real-valued queries.
    * **Gaussian Mechanism:** For adding noise to real-valued queries, often preferred for achieving (ε, δ)-differential privacy.
    * **Exponential Mechanism:** For privately selecting an item from a set of candidates.
    * **Report Noisy Max:** For privately releasing the index of the maximum value in a set of counts.
* **Composable:** Mechanisms can be chained together, allowing for complex analyses while maintaining accurate privacy accounting.
* **Rust-based:**  Benefits from Rust's performance, memory safety, and strong type system.
* **Clear API:** Designed for ease of use and integration into existing Rust projects.
* **Testing and Examples:** Includes comprehensive tests and examples to demonstrate usage and verify correctness.

## Getting Started

Add this library to your `Cargo.toml`:

```toml
[dependencies]
differential-privacy = "0.1.0"
