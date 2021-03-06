import json
import random
import warnings
from itertools import chain

from tqdm import tqdm, trange
from model.config import params_setup
from model.dataset import ATTR_TO_SPECIAL_TOKEN, SPECIAL_TOKENS
from model.dataset import FacebookPersonaDataset as fbdataset

import sys
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, DistributedSampler, SequentialSampler
from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                 GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)
from transformers import get_linear_schedule_with_warmup

history = []

def add_special_tokens(model,tokenizer):
  origin_num_tokens = len(tokenizer.encoder)
  num_special_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
  if num_special_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=origin_num_tokens + num_special_tokens)


def top_filtering(logits, top_k = 0, top_p = 0.9,threshold = -float('Inf'),filter_value=-float('Inf')):
  """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            threshold: a minimal threshold to keep logits
  """
  assert logits.dim() == 1 #batch_size = 1
  top_k = min(top_k,logits.size(-1))
  if top_k > 0:
    # Remove all tokens with a probability less than the last token in the top-k tokens
    indices_to_remove = logits < torch.top_k(logits,top_k)[0][...,-1,None]
    logits[indices_to_remove] = filter_value
  
  if top_p > 0.0:
    # Compute cumulative probabilities of sorted tokens
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probabilities > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Back to unsorted indices and set them to -infinity
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
  
  indices_to_remove = logits < threshold
  logits[indices_to_remove] = filter_value

  return logits

def sample_sequence(args,personality,history,tokenizer,model,current_output = None):
  """
    Generate reponse from previous reponses
  """
  special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
  if current_output is None:
    current_output = []
  for i in range (args.max_len):
    instance = fbdataset(args.url,args.dataset_cache_path,args,tokenizer).build_input(persona=personality,history=history,reply=current_output,with_eos=False)
    #print(tokenizer.decode(chain(*instance["input_ids"])),file=sys.stderr)
    input_ids = torch.tensor(instance["input_ids"],device= args.device).unsqueeze(0)
    token_type_ids = torch.tensor(instance["token_type_ids"], device= args.device).unsqueeze(0)
    
    logits = model(input_ids,token_type_ids = token_type_ids)
    if isinstance(logits, tuple):  # for gpt2 and maybe others
      logits = logits[0] 
    # logits shape (batch_size, num_choices, sequence_length, vocab_size)
    logits = logits[0,-1,:]/args.temperature 
    logits = top_filtering(logits,top_k = args.top_k,top_p = args.top_p)
    probs = F.softmax(logits, -1)

    prev = torch.topk(probs,1)[1] if args.no_sample else torch.multinomial(probs,1)
    
    if i < args.min_len and prev.item() in special_tokens_ids:
      while prev.item() in special_tokens_ids:
        if probs.max().item() == 1:
          warnings.warn("Warning: model generating special token with probability 1.")
          break  # avoid infinitely looping over special token
      prev = torch.multinomial(probs, num_samples=1)

    if prev.item() in special_tokens_ids:
      break
    current_output.append(prev.item())
  return current_output

def load_model(args):
  #reload checkpoints and evaluate on test dataset
  model_class_name = GPT2DoubleHeadsModel if "gpt2" in args.model_checkpoint else OpenAIGPTDoubleHeadsModel
  tokenizer_class_name = GPT2Tokenizer if "gpt2" in args.model_checkpoint else OpenAIGPTTokenizer

  model = model_class_name.from_pretrained(args.saved_dir)
  tokenizer =  tokenizer_class_name.from_pretrained(args.saved_dir)
  model.to(args.device)
  add_special_tokens(model,tokenizer)
  return model,tokenizer






  

  