import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import torch
from difflib import SequenceMatcher
import re


def load_finetuned_model(model_dir: str = "./finetuned_model"):
    """Load the finetuned model and tokenizer from directory."""
    print(f"Loading finetuned model from {model_dir}...")

    # Load PEFT model
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✓ Model loaded successfully")
    return model, tokenizer


def generate_answer(model, tokenizer, query: str, max_length: int = 512):
    """Generate answer for a given query."""
    prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{query}

### Response:
"""

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.95,
            num_beams=1,
        )

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the response part (after "### Response:")
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    return response


def load_training_data(output_folder: str = "output"):
    """Load training data from output folder (same as in finetune.py)."""
    from finetune import load_function_data, create_training_data

    functions_data = load_function_data(output_folder)
    training_examples = create_training_data(functions_data)
    return training_examples


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using sequence matching."""
    # Normalize texts
    text1_normalized = text1.lower()
    text2_normalized = text2.lower()

    # Remove extra whitespace
    text1_normalized = " ".join(text1_normalized.split())
    text2_normalized = " ".join(text2_normalized.split())

    # Calculate similarity ratio
    matcher = SequenceMatcher(None, text1_normalized, text2_normalized)
    return matcher.ratio()


def extract_function_names(text: str) -> set:
    """Extract function names from response text."""
    # Look for patterns like function names in markdown code blocks or plain text
    function_names = set()

    # Pattern 1: **function_name**
    pattern1 = r'\*\*(\w+)\*\*'
    matches = re.findall(pattern1, text)
    function_names.update(matches)

    # Pattern 2: `function_name`
    pattern2 = r'`(\w+)`'
    matches = re.findall(pattern2, text)
    function_names.update(matches)

    return function_names


def evaluate_answers(model, tokenizer, training_examples, top_k: int = 3):
    """Evaluate model answers and calculate accuracy metrics."""
    print(f"\nEvaluating {len(training_examples)} examples...\n")

    results = []
    correct = 0
    total = len(training_examples)

    for i, example in enumerate(training_examples):
        query = example["instruction"]
        expected_output = example["output"]

        print(f"[{i+1}/{total}] Query: {query[:50]}...")

        # Generate answer
        generated_answer = generate_answer(model, tokenizer, query)

        # Extract function names from expected and generated outputs
        expected_functions = extract_function_names(expected_output)
        generated_functions = extract_function_names(generated_answer)

        # Calculate overlap
        if expected_functions:
            overlap = len(expected_functions & generated_functions)
            precision = overlap / len(generated_functions) if generated_functions else 0
            recall = overlap / len(expected_functions) if expected_functions else 0
        else:
            precision = 0
            recall = 0

        # Calculate text similarity
        similarity = calculate_similarity(generated_answer, expected_output)

        # Consider it correct if similarity > 0.5 or recall > 0.5
        is_correct = similarity > 0.5 or recall > 0.5
        if is_correct:
            correct += 1

        results.append({
            "query": query,
            "expected_functions": list(expected_functions),
            "generated_functions": list(generated_functions),
            "similarity": similarity,
            "precision": precision,
            "recall": recall,
            "is_correct": is_correct,
        })

        print(f"  Expected functions: {expected_functions}")
        print(f"  Generated functions: {generated_functions}")
        print(f"  Similarity: {similarity:.2%}")
        print(f"  Correct: {'✓' if is_correct else '✗'}\n")

    # Calculate overall metrics
    accuracy = correct / total if total > 0 else 0
    avg_similarity = sum(r["similarity"] for r in results) / total if total > 0 else 0
    avg_precision = sum(r["precision"] for r in results) / total if total > 0 else 0
    avg_recall = sum(r["recall"] for r in results) / total if total > 0 else 0

    # Print summary
    print("="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Examples: {total}")
    print(f"Correct Answers: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average Similarity: {avg_similarity:.2%}")
    print(f"Average Precision: {avg_precision:.2%}")
    print(f"Average Recall: {avg_recall:.2%}")
    print("="*60)

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_similarity": avg_similarity,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "results": results,
    }


def main():
    # Load finetuned model
    model, tokenizer = load_finetuned_model("./finetuned_model")

    # Load training data
    print("\nLoading training data...")
    training_examples = load_training_data("output")
    print(f"Loaded {len(training_examples)} training examples")

    # Evaluate
    metrics = evaluate_answers(model, tokenizer, training_examples)

    # Save results to file
    output_file = "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()