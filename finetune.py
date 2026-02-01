import json
import os
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch


def load_function_data(output_folder: str):
    """Load function data from JSON files in output folder."""
    functions_data = []
    output_path = Path(output_folder)

    for json_file in output_path.glob("*.json"):
        with open(json_file, 'r') as f:
            kg_data = json.load(f)

        project_name = kg_data["name"]
        project_version = kg_data["version"]

        for code_file in kg_data.get("code", []):
            for func in code_file.get("functions", []):
                functions_data.append({
                    "project": f"{project_name} v{project_version}",
                    "file": code_file["path"],
                    "name": func["name"],
                    "signature": func["signature"],
                    "code": func["code"],
                    "description": func.get("description", ""),
                })

    return functions_data


def create_training_data(functions_data):
    """Create training examples for function retrieval."""
    training_examples = []

    # Example queries that should match ed25519 functions
    queries = [
        "generate and sign a message using ed25519",
        "how to create ed25519 signature",
        "generate cryptographic key pair",
        "verify ed25519 signature",
        "sign data with private key",
        "create ed25519 keys",
    ]

    for query in queries:
        # Find relevant functions based on keywords
        relevant_functions = []
        query_lower = query.lower()

        for func in functions_data:
            # Check if function is relevant
            is_relevant = False
            if any(keyword in func["name"].lower() for keyword in ["key", "sign", "verify", "generate"]):
                is_relevant = True
            if any(keyword in func["description"].lower() for keyword in ["ed25519", "signature", "key", "sign"]):
                is_relevant = True

            if is_relevant:
                relevant_functions.append(func)

        # Create training example
        if relevant_functions:
            # Format as Q&A
            answer = f"To {query}, you can use the following functions:\n\n"
            for func in relevant_functions[:3]:  # Limit to top 3 functions
                answer += f"**{func['name']}**\n"
                answer += f"Signature: `{func['signature']}`\n"
                answer += f"File: {func['file']}\n"
                answer += f"Description: {func['description'][:200]}...\n\n"

            training_examples.append({
                "instruction": query,
                "output": answer.strip(),
            })

    for example in training_examples:
        print(f"Instruction: {example['instruction']}\nOutput: {example['output']}\n{'-'*40}")

    return training_examples


def prepare_dataset(training_examples, tokenizer):
    """Prepare dataset for training."""
    def format_prompt(example):
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Response:
{example['output']}"""
        return {"text": prompt}

    # Convert to dataset
    dataset = Dataset.from_list(training_examples)
    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

    # Tokenize
    def tokenize_function(examples):
        # Handle both single and batched inputs
        texts = examples["text"] if isinstance(examples["text"], list) else [examples["text"]]

        outputs = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=2048,
            return_tensors=None,  # Return lists, not tensors
        )
        # Set labels (copy of input_ids, with padding tokens masked)
        outputs["labels"] = [ids[:] for ids in outputs["input_ids"]]
        return outputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset


def setup_lora_model(model_name: str, use_cpu: bool = True):
    """Setup model with LoRA configuration."""

    if use_cpu:
        # Load model without quantization for CPU training
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
    else:
        # Quantization config for efficient GPU training
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Determine device
        device_map = {"": torch.cuda.current_device()}

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )

        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def finetune_model(
    model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    output_folder: str = "output",
    output_dir: str = "./finetuned_model",
):
    """Fine-tune LLM model with LoRA for function retrieval."""

    print("Loading function data...")
    functions_data = load_function_data(output_folder)
    print(f"Loaded {len(functions_data)} functions")

    print("\nCreating training data...")
    training_examples = create_training_data(functions_data)
    print(f"Created {len(training_examples)} training examples")

    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"  # Important for causal LM

    model = setup_lora_model(model_name, use_cpu=True)
    model.config.pad_token_id = tokenizer.pad_token_id

    print("\nPreparing dataset...")
    tokenized_dataset = prepare_dataset(training_examples, tokenizer)

    print("\nSetting up training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=False,  # Disabled for CPU
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        warmup_steps=10,
        optim="adamw_torch",  # Use standard optimizer for CPU
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        use_cpu=True,  # Force CPU training
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    print("\nStarting training...")
    trainer.train()

    print("\nSaving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n✓ Model saved to {output_dir}")


def main():
    finetune_model(
        model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        output_folder="output",
        output_dir="./finetuned_model",
    )


if __name__ == "__main__":
    main()