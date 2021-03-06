import torch
from argparse import ArgumentParser
import os
import sys
from pathlib import Path
#define hyperparameters
def params_setup():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="gpt", help="short name of the model") 
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="") 
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Num of candidates") 
    parser.add_argument("--num_candidates", type=int, default=2, help="Num of candidates") #4
    parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
    parser.add_argument("--num_history", type=int, default=2, help="Number of previous exchanges to keep in history")
    parser.add_argument("--fp16_training", type=str, default="O1", help="Set to O0, O1, O2 or O3 for fp16 training")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--no_sample", type=bool, default=False, help="")
    parser.add_argument("--lm_coef", type=float, default=2.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering))") 
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)") 
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling softmax temperature") 
    parser.add_argument("--max_len", type=int, default=20, help="Maximum length of the output utterances") 
    parser.add_argument("--min_len", type=int, default=1, help="Min len of the output utterances")
    parser.add_argument("--num_gpu", type=int, default=torch.cuda.device_count(), help="Number of GPUS") #1
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Accumulate gradients on several steps")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--longest_common", type=int, default=5, help="Long cmommon")
    #path
    parser.add_argument("--url", type=str, default="https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json", help="Path or url of the dataset.")
    parser.add_argument("--local_path",type=str, default="", help="Path of the dataset on local.")
    parser.add_argument("--dataset_cache_path", type=str, default="", help="Path or url of the dataset cache")
    parser.add_argument("--saved_dir", type=str, default="./checkpoint/", help="path or url of checkpoint")
    parser.add_argument("--persona_path", type=str, default="./model/personality.txt", help="path or url of personality")
    args, _ = parser.parse_known_args()
    return args

def load_personality(args):
    if args.persona_path and os.path.isfile(args.persona_path):
        lines = open(args.persona_path).read().split('\n')[:-1]
        lines = [line.lower() for line in lines]
        return lines
    else:
        return []
