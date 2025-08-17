import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# --- Configuration ---
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'finetune', 'finetune_data.jsonl')

def train_lora_model():
    """Fine-tunes a model using the LoRA method."""

    print("Loading dataset...")
    dataset = load_dataset('json', data_files=DATASET_PATH, encoding='latin-1')

    # 1. Quantization configuration for loading in 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    # 2. Load the model and tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        fp16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard"
    )

    # 5. Initialize the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    # 6. Start training
    print("Starting training...")
    trainer.train()

    # Save the fine-tuned model (LoRA adapters)
    print("Training complete. Saving model adapters...")
    trainer.save_model("finetuned_model")
    print("Model adapters saved to 'finetuned_model' directory.")

if __name__ == "__main__":
    train_lora_model()
