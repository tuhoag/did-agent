use crypto;

fn main() {
    let (public_key, private_key) = crypto::generate_ed25519_key_pair();
    let data = b"Hello, world!";
    let signature = crypto::generate_ed25519_signature(data, &private_key);
    let is_valid = crypto::verify_ed25519_signature(data, &signature, &public_key);
    println!("Signature valid: {}", is_valid);
}