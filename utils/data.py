from datasets import load_dataset
import numpy as np
import torch
import json
from torch.utils.data import Dataset
from transformers.modeling_utils import PreTrainedModel
from typing import Any, List, Tuple
import random


class TextDataset:
    def __init__(self, device, dataset, split, n_inputs, batch_size, cache_dir=None):
        
        seq_keys = {
            'cola': 'sentence',
            'sst2': 'sentence',
            'rte': 'sentence1',
            'rotten_tomatoes': 'text',
            'glnmario/ECHR': 'text',
            'stanfordnlp/imdb': 'text'
        }
        seq_key = seq_keys[dataset]

        if dataset in ['cola', 'sst2', 'rte']:
            full = load_dataset('glue', dataset, cache_dir=cache_dir)['train']
        elif dataset == 'glnmario/ECHR':
            full = load_dataset('csv', data_files = ['models_cache/datasets--glnmario--ECHR/ECHR_Dataset.csv'], cache_dir=cache_dir)['train']
        else:
            full = load_dataset(dataset)['train']
            
        idxs = list(range(len(full)))
        np.random.shuffle(idxs)
        #if dataset == 'cola':
        #    import pdb; pdb.set_trace()
        #    assert idxs[0] == 2310 # with seed 101

        n_samples = n_inputs * batch_size

        if split == 'test':
            assert n_samples <= 1000
            idxs = idxs[:n_samples]
        elif split == 'val':
            idxs = idxs[1000:] # first 1000 saved for testing
            
            final_idxs = []
            while len(final_idxs) < n_samples:
                zipped = [(idx, len(full[idx][seq_key])) for idx in idxs]
                zipped = sorted(zipped, key=lambda x: x[1])
                chunk_sz = max(len(zipped) // n_samples, 1) 
                
                l = min(len(zipped), n_samples - len(final_idxs))
                for i in range(l):
                    tmp = chunk_sz*i + np.random.randint(0, chunk_sz)
                    final_idxs.append(zipped[tmp][0])
                np.random.shuffle(idxs) 
                
            np.random.shuffle(final_idxs) 
            idxs = final_idxs
        
        # Slice
        self.seqs = []
        self.labels = []
        for i in range(n_inputs):
            seqs = []
            for j in range(batch_size):
                seqs.append(full[idxs[i*batch_size+j]][seq_key])
            if dataset != 'glnmario/ECHR':
                labels = torch.tensor([full[idxs[i*batch_size : (i+1)*batch_size]]['label']], device=device)
            else:
                labels = torch.tensor([full[idxs[i*batch_size : (i+1)*batch_size]]['binary_judgement']], device=device)
            self.seqs.append(seqs)
            self.labels.append(labels)
        assert len(self.seqs) == n_inputs
        assert len(self.labels) == n_inputs


    def __getitem__(self, idx):
        return (self.seqs[idx], self.labels[idx])


class DonutDataset(Dataset):
    """
    DonutDataset which is saved in huggingface datasets format. (see details in https://huggingface.co/docs/datasets)
    Each row, consists of image path(png/jpg/jpeg) and gt data (json/jsonl/txt),
    and it will be converted into input_tensor(vectorized image) and input_ids(tokenized string).
    Args:
        dataset_name_or_path: name of dataset (available at huggingface.co/datasets) or the path containing image files and metadata.jsonl
        max_length: the max number of tokens for the target sequences
        split: whether to load "train", "validation" or "test" split
        ignore_id: ignore_index for torch.nn.CrossEntropyLoss
        task_start_token: the special token to be fed to the decoder to conduct the target task
        prompt_end_token: the special token at the end of the sequences
        sort_json_key: whether or not to sort the JSON keys
    """

    def __init__(
            self,
            dataset_name_or_path: str,
            max_length: int,
            split: str = "train",
            ignore_id: int = -100,
            task_start_token: str = "<s>",
            prompt_end_token: str = None,
            sort_json_key: bool = True,
            processor=None,
            model=PreTrainedModel,
            **kwargs,
    ):
        super().__init__()
        self.model = model.module if type(model) is torch.nn.parallel.DataParallel else model
        self.processor = processor

        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
        self.sort_json_key = sort_json_key

        self.dataset = load_dataset(dataset_name_or_path, split=self.split)
        # load_dataset('/home/zolfi/GenAI/datasets/docvqa_new')
        # self.dataset = load_dataset(dataset_name_or_path)
        self.dataset_length = len(self.dataset)

        self.added_tokens = []
        # additional_tokens = ["", ""]
        #
        # self.add_tokens(additional_tokens)

        self.gt_token_sequences = []
        for sample in self.dataset:
            ground_truth = json.loads(sample["ground_truth"])
            if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
                assert isinstance(ground_truth["gt_parses"], list)
                gt_jsons = ground_truth["gt_parses"]
            else:
                assert "gt_parse" in ground_truth and isinstance(ground_truth["gt_parse"], dict)
                gt_jsons = [ground_truth["gt_parse"]]

            self.gt_token_sequences.append(
                [
                    self.task_start_token +
                    self.json2token(
                        gt_json,
                        update_special_tokens_for_json_key=self.split == "train",
                        sort_json_key=self.sort_json_key,
                    )
                    + processor.tokenizer.eos_token
                    for gt_json in gt_jsons  # load json from list of json
                ]
            )

        self.add_tokens([self.task_start_token, self.prompt_end_token])
        self.prompt_end_token_id = processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    # self.donut_model.decoder.add_special_tokens([self.task_start_token, self.prompt_end_token])
        # self.prompt_end_token_id = self.donut_model.decoder.tokenizer.convert_tokens_to_ids(self.prompt_end_token)

    def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.add_tokens([fr"<s_{k}>", fr"</s_{k}>"])
                    output += (
                            fr"<s_{k}>"
                            + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                            + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in self.added_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj

    def add_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings of the decoder
        """
        newly_added_num = self.processor.tokenizer.add_tokens(list_of_tokens)
        if newly_added_num > 0:
            self.model.decoder.resize_token_embeddings(len(self.processor.tokenizer))
            self.added_tokens.extend(list_of_tokens)

    def __len__(self) -> int:
        return self.dataset_length - 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load image from image_path of given dataset_path and convert into input_tensor and labels
        Convert gt data into input_ids (tokenized string)
        Returns:
            input_tensor : preprocessed image
            input_ids : tokenized gt_data
            labels : masked labels (model doesn't need to predict prompt and pad token)
        """
        sample = self.dataset[idx]

        # input_tensor
        pixel_values = self.processor(sample["image"].convert("RGB"), random_padding=self.split == "train", return_tensors="pt").pixel_values
        input_tensor = pixel_values.squeeze()

        # input_ids
        processed_parse = random.choice(self.gt_token_sequences[idx])  # can be more than one, e.g., DocVQA Task 1
        tokenizer_results = self.processor.tokenizer(
            processed_parse,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokenizer_results["input_ids"].squeeze(0)
        attention_mask = tokenizer_results["attention_mask"]

        if self.split == "train":
            labels = input_ids.clone()
            labels[
                labels == self.processor.tokenizer.pad_token_id
            ] = self.ignore_id  # model doesn't need to predict pad token
            labels[
                : torch.nonzero(labels == self.prompt_end_token_id).sum() + 1
            ] = self.ignore_id  # model doesn't need to predict prompt (for VQA)
            return input_tensor, input_ids, labels, attention_mask
        else:
            prompt_end_index = torch.nonzero(
                input_ids == self.prompt_end_token_id
            ).sum()  # return prompt end index instead of target output labels
            return input_tensor, input_ids, prompt_end_index, processed_parse