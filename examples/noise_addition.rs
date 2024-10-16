// examples/noise_addition.rs

use differential_privacy::mechanisms::laplace_mechanism;
use differential_privacy::privacy_accounting::PrivacyAccountant;

fn main() {
    let value = 100.0;
    let sensitivity = 1.0;
    let epsilon = 0.5;
    let mut accountant = PrivacyAccountant::new(0.0, 0.0);
    let noisy_value = laplace_mechanism(value, sensitivity, epsilon, &mut accountant);
    println!("Original Value: {}", value);
    println!("Noisy Value: {}", noisy_value);
    let (total_epsilon, total_delta) = accountant.get_privacy_loss();
    println!("Total Epsilon: {}", total_epsilon);
    println!("Total Delta: {}", total_delta);
}