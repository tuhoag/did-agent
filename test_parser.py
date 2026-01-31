from tree_sitter import Language, Parser, Query, QueryCursor
import tree_sitter_rust as rpython

from agents.code_extractor_agent import FunctionExtractor

def main():
    print("Hello from did-agent!")
    code = """use ed25519_compact::{KeyPair, Noise, PublicKey, Signature};
            pub fn generate_key_pair() -> (Vec<u8>, Vec<u8>) {
                let key_pair = KeyPair::from_seed(Seed::default());
                let public_key = key_pair.pk.to_vec();
                let private_key = key_pair.sk.to_vec();
                (public_key, private_key)
            }

            pub fn generate_signature(data: &[u8], private_key: &[u8]) -> Vec<u8> {
                let sk = ed25519_compact::SecretKey::from_slice(private_key).expect("Invalid private key");
                let key_pair = KeyPair::from_sk(&sk);
                key_pair.sk.sign(data, Some(Noise::default())).to_vec()
            }

            pub fn verify_signature(data: &[u8], signature: &[u8], public_key: &[u8]) -> bool {
                let pk = PublicKey::from_slice(public_key).ok();
                let sig = Signature::from_slice(signature).ok();

                match (pk, sig) {
                    (Some(pk), Some(sig)) => pk.verify(data, &sig).is_ok(),
                    _ => false,
                }
            }

            fn main() {
                let (public_key, private_key) = generate_key_pair();
                let data = b"Hello, world!";
                let signature = generate_signature(data, &private_key);
                let is_valid = verify_signature(data, &signature, &public_key);
                println!("Signature valid: {}", is_valid);
            }
        """

    extractor = FunctionExtractor()
    result = extractor.extract_rust_functions(code)
    print("Extracted functions:")
    print(result)

    # # Setup
    # R_LANGUAGE = Language(rpython.language())
    # parser = Parser(R_LANGUAGE)

    # tree = parser.parse(bytes(code, "utf8"))
    # root_node = tree.root_node
    # # print(f"Root node: {root_node}")
    # # query = Query(R_LANGUAGE, """
    # # (function_item
    # #     (visibility_modifier)
    # #     name: (identifier) @function.name
    # # ) @function.definition
    # # """)
    # # cursor = QueryCursor(query)
    # # for match in cursor.matches(root_node):
    # #     print(f"Match: {match}")


    # for child in root_node.children:
    #     if child.type == "function_item":
    #         print("Function found:")
    #         print(code[child.start_byte:child.end_byte])
    #         print("-----")
    #         print(child)


if __name__ == "__main__":
    main()