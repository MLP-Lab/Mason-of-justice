from load_model import load_model
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import os

class SaveCallback(TrainerCallback):
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of training.
        """
        save_path = os.path.join(args.output_dir, "pretrained_lora")
        kwargs['model'].save_pretrained(save_path)
        kwargs['tokenizer'].save_pretrained(save_path)
        print(f"Model and Tokenizer saved at {save_path}")

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """
        if state.best_model_checkpoint is not None:
            checkpoint_path = os.path.join(state.best_model_checkpoint, "pretrained_lora")
        else:
            checkpoint_path = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        
        save_path = os.path.join(checkpoint_path, "pretrained_lora")
        kwargs['model'].save_pretrained(save_path)
        kwargs['tokenizer'].save_pretrained(save_path)
        print(f"Model and Tokenizer saved at {save_path}")
        return control

def accuracy(predictions, references, normalize=True, sample_weight=None):
        return {
            "accuracy": float(
                accuracy_score(references, predictions, normalize=normalize, sample_weight=sample_weight)
            )
        }

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = labels[:, 1:].reshape(-1)
    preds = preds[:, :-1].reshape(-1)
    return accuracy(predictions=preds, references=labels)

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)