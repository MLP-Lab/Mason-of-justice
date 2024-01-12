from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets
import torch
import glob
from tqdm.auto import tqdm

for i in range(12):
    file = glob.glob(f'./backup/test/*{i}.pt')[0]
    num = file.split('_')[2].split('.')[0].strip()
    print(f'{num}: loading by torch.load...')
    temp_input_ids_dict = {
        'input_ids': torch.load(file)
    }
    print(f'{num}: Complete load by torch.load!')

    print(f'{num}: Making Dataset by Dataset.from_dict...')
    dataset = Dataset.from_dict(temp_input_ids_dict)
    print(f'{num}: Complete Dataset by Dataset.from_dict!')

    print(f'{num}: Delete Variable to save the memory!')
    del temp_input_ids_dict

    print(f'{num}: Saving Dataset by .save_to_disk...')
    dataset.save_to_disk(f'/home/mjkim/kimwon/data/pt_data_input_ids/pt_data_input_ids_{num}/', num_proc=16)
    print(f'{num}: Complete Save by .save_to_disk!')

    print(f'{num}: Delete Variables to save the memory!')
    del dataset