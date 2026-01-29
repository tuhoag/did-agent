import time

from agents.model_factory import Model
from langchain_community.llms import VLLM
from langchain.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage

import logging

logging.getLogger("vllm").setLevel(logging.ERROR)


def inference_with_ollama():
    start_time = time.time()
    model_name = "qwen2.5-coder:0.5b"
    llm = Model(model_name, backend="ollama", temperature=0.7)
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    response = llm.invoke(messages)

    end_time = time.time()
    print(f"Ollama Inference Time: {end_time - start_time} seconds")

def inference_with_vllm():
    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    llm = Model(model_name, backend="vllm", temperature=0.7)
    # llm = VLLM(model=model_name)
    # params = SamplingParams(temperature=0.7, top_p=0.95)

    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    start_time = time.time()
    # response = llm.invoke(messages)
    response = llm.invoke(messages)

    end_time = time.time()
    print(f"VLLM Inference Time: {end_time - start_time} seconds")

def extract_code() -> str:
    model_name = "Qwen/Qwen2.5-Coder-1.5B"
    backend = "vllm"
    llm = Model(model_name, backend=backend, temperature=0.0)

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

    extract_system = SystemMessage(
            content=(
                """You are a code analysis agent. Extract ALL function definitions 
                from the given code, preserving arguments and return types if present. 
                Return the result as a JSON array of strings, where each string contains one function's code.                 
                
                Schema:                
                [
                    "function code 1",
                    "function code 2",
                ]
                
                Requirements:
                - Return ONLY the function code strings in a JSON array.
                - No additional text outside the JSON array.
                - Preserve function signatures, including argument names and types, and return types.
                """
            )
        )
    response = llm.invoke([extract_system, HumanMessage(content=code)])
    print(f"Extraction response: {response}")
    # return code

def main():
    # print("This is the main function of vllm.py")

    # inference_with_ollama()
    # inference_with_vllm()

    extract_code()



if __name__ == "__main__":
    main()