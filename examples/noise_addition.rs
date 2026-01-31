// examples/noise_addition.rs

use differential_privacy::mechanisms::{
    laplace_mechanism, gaussian_mechanism, exponential_mechanism, report_noisy_max,
};
use differential_privacy::privacy_accounting::{PrivacyAccountant, CompositionMethod};

fn main() {
    println!("=== Differential Privacy Library Demo ===\n");

    // Create a privacy accountant to track privacy budget
    let mut accountant = PrivacyAccountant::new();

    // --- Laplace Mechanism ---
    println!("1. Laplace Mechanism");
    let value = 100.0;
    let sensitivity = 1.0;
    let epsilon = 0.5;
    let noisy_value = laplace_mechanism(value, sensitivity, epsilon, &mut accountant);
    println!("   Original Value: {}", value);
    println!("   Noisy Value: {:.2}", noisy_value);

    // --- Gaussian Mechanism ---
    println!("\n2. Gaussian Mechanism");
    let delta = 1e-5;
    let noisy_value = gaussian_mechanism(value, sensitivity, epsilon, delta, &mut accountant)
        .expect("Invalid parameters");
    println!("   Original Value: {}", value);
    println!("   Noisy Value: {:.2}", noisy_value);

    // --- Exponential Mechanism ---
    println!("\n3. Exponential Mechanism");
    let utilities = vec![10.0, 25.0, 15.0, 5.0];
    let selected = exponential_mechanism(&utilities, sensitivity, epsilon, &mut accountant)
        .expect("Invalid parameters");
    println!("   Utilities: {:?}", utilities);
    println!("   Selected Index: {} (utility: {})", selected, utilities[selected]);

    // --- Report Noisy Max ---
    println!("\n4. Report Noisy Max");
    let counts = vec![150.0, 200.0, 175.0, 50.0];
    let (winner, noisy_max) = report_noisy_max(&counts, sensitivity, epsilon, &mut accountant)
        .expect("Invalid parameters");
    println!("   Counts: {:?}", counts);
    println!("   Winner Index: {} (noisy count: {:.2})", winner, noisy_max);

    // --- Privacy Accounting ---
    println!("\n=== Privacy Composition Summary ===");
    let summary = accountant.composition_summary(1e-5);
    print!("{}", summary);

    // Compare composition methods
    println!("\n=== Composition Method Comparison ===");
    let (basic_eps, basic_delta) = accountant.get_privacy_loss(CompositionMethod::Basic);
    let (adv_eps, adv_delta) = accountant.get_privacy_loss(CompositionMethod::Advanced);

    println!("Basic composition:    ε = {:.4}, δ = {:.2e}", basic_eps, basic_delta);
    println!("Advanced composition: ε = {:.4}, δ = {:.2e}", adv_eps, adv_delta);

    if adv_eps < basic_eps {
        let improvement = (1.0 - adv_eps / basic_eps) * 100.0;
        println!("Advanced composition saves {:.1}% privacy budget!", improvement);
    }
}
