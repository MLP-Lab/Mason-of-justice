from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import os
from torch.utils.data import Dataset, IterableDataset
from sklearn.metrics import accuracy_score
from collections import deque
import torch
from tqdm.auto import tqdm
import glob
from datasets import load_from_disk, concatenate_datasets

class CustomDataset(Dataset):
    def __init__(self, files_dir, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.all_datasets = []

        for file_dir in files_dir:
            self.all_datasets.append(load_from_disk(file_dir))

        self.all_datasets = concatenate_datasets(self.all_datasets)
        self.all_datasets = self.all_datasets.shuffle(seed=42)
        # test
        # self.all_datasets = self.all_datasets.train_test_split(test_size=10)['test']

    def __len__(self):
        return len(self.all_datasets)
    
    def __getitem__(self, idx):
        if len(self.all_datasets[idx]['input_ids']) > self.max_sequence_length:
            return {'input_ids': self.all_datasets[idx]['input_ids'][:self.max_sequence_length]}
        else:
            return {'input_ids': self.all_datasets[idx]['input_ids']}

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


# class CustomDataset(Dataset):
#     def __init__(self, tokenizer, max_sequence_length, characters_per_token=3.5):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.max_sequence_length = max_sequence_length
#         self.input_ids = []
#         self.max_char_length = int(max_sequence_length * characters_per_token)
#         # self.cnt = 1
#         # raw_dataset.map(lambda x: self._prepare_dataset_for_max_sequence(x), batched=True, keep_in_memory=False)
#         # torch.save(self.input_ids, f'/home/mjkim/kimwon/backup/test/input_ids_{self.cnt}.pt')
#         # self.input_ids = []

#     def _prepare_dataset_for_max_sequence(self, samples):
#         # print(len(self.input_ids))
#         # if len(self.input_ids) > 530000:
#         #     torch.save(self.input_ids, f'/home/mjkim/kimwon/backup/test/input_ids_{self.cnt}.pt')
#         #     self.input_ids = []
#         #     self.cnt += 1

#         for sample in samples['text']:
#             if len(sample) <= self.max_char_length:
#                 output = self.tokenizer(sample, return_attention_mask=False, truncation=True, max_length=self.max_sequence_length)
#                 self.input_ids.append(output['input_ids'])
            
#             else:
#                 _divided_num = len(sample) // self.max_char_length
#                 for i in range(_divided_num + 1):
#                     if i == _divided_num:
#                         output = self.tokenizer(sample[self.max_char_length * i : ], return_attention_mask=False, truncation=True, max_length=self.max_sequence_length)
#                         self.input_ids.append(output['input_ids'])
#                     else:
#                         output = self.tokenizer(sample[self.max_char_length * i : self.max_char_length * (i+1)], return_attention_mask=False, truncation=True, max_length=self.max_sequence_length)
#                         self.input_ids.append(output['input_ids'])

#     def __len__(self):
#         return len(self.input_ids)
    
#     def __getitem__(self, idx):
#         assert len(self.input_ids[idx]) <= self.max_sequence_length, ValueError('input_ids is longer than max_sequence_length')
#         return {'input_ids': self.input_ids[idx]}