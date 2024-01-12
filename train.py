from models.load_model import load_model
from utils.utils import CustomDataset, SaveCallback
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import load_from_disk
import os
import glob


def main():
    # Set model & tokenizer
    model, tokenizer = load_model(model_name='meta-llama/Llama-2-7b-hf')
    tokenizer.pad_token = tokenizer.eos_token

    # Train Dataset
    data_files_dir = glob.glob('/data/MLP/ihwon/data/*')
    train_dataset = CustomDataset(files_dir=data_files_dir, max_sequence_length=4096)
    
    # Data Collator for Causal Language Modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir = '/data/MLP/ihwon/Mason_of_justice/checkpoints',
        do_train=True,
        per_device_train_batch_size=1,
        learning_rate=2e-5,
        warmup_ratio=0.05,
        weight_decay=0.01,
        num_train_epochs=1,
        save_strategy='steps',
        logging_steps=50,
        save_steps=2000,
        # bf16=True,
        deepspeed='/data/MLP/ihwon/Mason_of_justice/ds_zero2_no_offload.json',
        save_total_limit=5,
        ddp_timeout=30000,
        logging_first_step=True,
        gradient_accumulation_steps=16,
        lr_scheduler_type='cosine',
        torch_compile=True,
        torch_compile_backend='inductor',
        torch_compile_mode='default',
        optim='adamw_torch',
        bf16=True,
    )   

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # trainer.add_callback(SaveCallback)
    # trainer.train(resume_from_checkpoint='/data/MLP/ihwon/Mason_of_justice/checkpoints/checkpoint-2000')
    trainer.train()



if __name__ == '__main__':
    main()