import torch
import json
import os
from collections import defaultdict
from itertools import chain
from torch.utils.data import DataLoader, TensorDataset

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

def tokenize(obj,tokenizer):
    if isinstance(obj,str):
      return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj,dict):
      return dict((n, tokenize(o,tokenizer)) for n, o in obj.items())
    return list(tokenize(o,tokenizer) for o in obj)

class FacebookPersonaDataset():
  """
    Concatenate context segments in a single sequence [[bos+persona], [history], [reply+eos]]
    Tokenize and convert them to tensor
    
  """
  def __init__(self,url,cache_path,args,tokenizer = ''):
    self.file_path = url
    self.tokenizer = tokenizer
    self.cache_path = cache_path
    self.args = args

  def load_dataset(self):
    #self.cache_path = self.cache_path + '_' + type(self.tokenizer).__name__  # To avoid using GPT cache for GPT-2 and vice-versa
    if self.cache_path and os.path.isfile(self.cache_path):
      #load tokenized dataset
      dataset = torch.load(self.cache_path)
      print("dataset loaded")
    else:
      with open(self.file_path, "r", encoding="utf-8") as f:
        dataset = json.loads(f.read())
      
      dataset = tokenize(dataset,self.tokenizer)
      torch.save(dataset,self.cache_path)
    return dataset

  def _pad_dataset(self,dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -100] * (max_l - len(x)) for x in dataset[name]]
    return dataset

  def build_input(self,persona,history,reply,lm_labels = False,with_eos = True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    #bos, eos, speaker1, speaker2 = "<bos>", "<eos>", "<speaker1>", "<speaker2>"
    #sequence = [[bos] + list(persona) + ["his"] + history + ["rep"] + [reply]]
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])] #add bos, eos
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])] #add speaker1, speaker2
    #after concat: [[bos+persona], [history], [reply+eos]]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    #segment
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    #position
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    #language model labels is used to calculate lm_loss
    instance["lm_labels"] = [-100] * len(instance["input_ids"]) #labels set to -100 are ignored (masked)
    if lm_labels:
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
    return instance

  def get_data_loaders(self):
    personachat_dataset = self.load_dataset()
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for name,dataset in personachat_dataset.items():
      num_candidates = len(dataset[0]["utterances"][0]["candidates"]) #num_candidates are same for all dialoges
      if self.args.num_candidates > 0 and name == 'train':
        num_candidates =  min(num_candidates ,self.args.num_candidates)
      for dialoge in dataset:
        persona = dialoge["personality"].copy()
        for _ in range(self.args.personality_permutations):
          for utterance in dialoge["utterances"]:
            history = utterance["history"][-(2*self.args.num_history+1):]
            for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
              #The last candidate is a ground truth reponse.
              lm_labels = bool(j==num_candidates-1)
              instance = self.build_input(persona,history,candidate,lm_labels)
              for input_name,data in instance.items():
                datasets[name][input_name].append(data) #datasets['train']['input_ids'] of [[c1 in u1],[c2 in u1],..,[c1 in u 7][c2 in u7]]
            datasets[name]["mc_labels"].append(num_candidates - 1) #7
            datasets[name]["n_candidates"] = num_candidates
          persona = [persona[-1]] + persona[:-1]  # permuted personalitie
          
    #pad input and convert to tensor
    print("pad input and convert to tensor")
    tensor_datasets = {"train": [], "valid": []}
    # for dataset_name, dataset in datasets.items():
    #   dataset =  self._pad_dataset(dataset, padding=self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
    #   for input_name in MODEL_INPUTS:
    #     tensor = torch.tensor(dataset[input_name])
    #     if input_name != "mc_labels":
    #       tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
    #     tensor_datasets[dataset_name].append(tensor)

    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    return train_dataset, valid_dataset 