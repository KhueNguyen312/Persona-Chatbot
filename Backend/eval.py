import random
import logging
from pprint import pformat
from collections import defaultdict
from functools import partial
from tqdm import trange
from itertools import chain

import torch
import torch.nn.functional as F
from parlai.core.agents import Agent
from parlai.scripts.eval_model import setup_args as base_setup_args
from projects.convai2.eval_hits import eval_hits, setup_args as setup_args_hits
from projects.convai2.eval_f1 import eval_f1, setup_args as setup_args_f1
from projects.convai2.eval_ppl import eval_ppl, setup_args as setup_args_ppl
from projects.convai2.build_dict import build_dict

from transformers import (OpenAIGPTDoubleHeadsModel, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                  GPT2DoubleHeadsModel, GPT2LMHeadModel, GPT2Tokenizer)

# from model.interact import add_special_tokens
# from model.dataset import SPECIAL_TOKENS
SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}

MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-100] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
    return instance

def add_special_tokens(model,tokenizer):
  origin_num_tokens = len(tokenizer.encoder)
  num_special_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
  if num_special_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=origin_num_tokens + num_special_tokens)


def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
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


def sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                   # warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -100] * (max_l - len(x)) for x in dataset[name]]
    return dataset

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class TransformerAgent(Agent):
    @staticmethod
    def add_cmdline_args(argparser):
        agent_args = argparser.add_argument_group('Agent parameters')
        agent_args.add_argument("--model_type", type=str, default="gpt", help="")
        agent_args.add_argument("--model_checkpoint", type=str, default="./data/", help="Path, url or short name of the model. Must be OpenAIGPT.")
        agent_args.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
        agent_args.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
        agent_args.add_argument("--eval_type", type=str, default="hits@1", help="hits@1, ppl or f1")
        agent_args.add_argument("--no_sample", action='store_true')
        agent_args.add_argument("--max_length", type=int, default=20)
        agent_args.add_argument("--min_length", type=int, default=1)
        agent_args.add_argument("--seed", type=int, default=0)
        agent_args.add_argument("--temperature", type=int, default=0.7)
        agent_args.add_argument("--top_k", type=int, default=0)
        agent_args.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
        return argparser
        
    def __init__(self, opt, shared=None):
        super(TransformerAgent, self).__init__(opt, shared)

        args = AttrDict(opt)  # to keep most commands identical to the interact.py script
        self.args = args

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__file__)
        self.logger.info(pformat(args))

        random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        if shared is None:
            self.logger.info("Get pretrained model and tokenizer")
            if args.model_type == "":
                print("checkpoint mustn't be empty")
            if 'gpt2' in args.model_type: 
                self.tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
                model_class = GPT2DoubleHeadsModel if self.args.eval_type == "hits@1" else GPT2LMHeadModel
            else:
                self.tokenizer = OpenAIGPTTokenizer.from_pretrained(args.model_checkpoint)
                model_class = OpenAIGPTDoubleHeadsModel if self.args.eval_type == "hits@1" else OpenAIGPTLMHeadModel

            self.model_checkpoint = model_class.from_pretrained(args.model_checkpoint)
            self.model_checkpoint.to(args.device)

            self.logger.info("Build BPE prefix dictionary")
            convai_dict = build_dict()
            assert len(convai_dict) == 19304
            self.prefix2words = self.get_prefix2words(convai_dict)
        else:
            self.model_checkpoint = shared['model']
            self.tokenizer = shared['tokenizer']
            self.prefix2words = shared['prefix2words']
        add_special_tokens(self.model_checkpoint, self.tokenizer)
        self.special_tokens_ids = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

        self.persona = []
        self.history = []
        self.labels = []

        self.reset()

    def observe(self, observation):
        if self.episode_done:
            self.reset()

        if self.labels:
            # Add the previous response to the history
            self.history.append(self.labels)

        if 'labels' in observation or 'eval_labels' in observation:
            text = observation.get('labels', observation.get('eval_labels', [[]]))[0]
            self.labels = self.tokenizer.encode(text)

        if 'text' in observation:
            text = observation['text']
            for subtext in text.split('\n'):
                subtext = subtext.strip()
                if subtext.startswith('your persona:'):
                    subtext = subtext.replace('your persona:', '').strip()
                    self.persona.append(self.tokenizer.encode(subtext))
                else:
                    self.history.append(self.tokenizer.encode(subtext))

        self.history = self.history[-(2*self.args.max_history+1):]

        candidates = []
        if 'label_candidates' in observation:
            for candidate in observation['label_candidates']:
                candidates.append((self.tokenizer.encode(candidate), candidate))
        self.candidates = candidates

        self.episode_done = observation['episode_done']
        self.observation = observation
        return observation

    def act(self):
        reply = {}

        if self.args.eval_type == "hits@1" and len(self.candidates) > 0:
            instances = defaultdict(list)
            for candidate, _ in self.candidates:
                instance = build_input_from_segments(self.persona, self.history, candidate, self.tokenizer)
                for input_name, input_array in instance.items():
                    instances[input_name].append(input_array)

            inputs = pad_dataset(instances, padding=self.special_tokens_ids[-1])

            tensor_inputs = {}
            for input_name in ["input_ids", "mc_token_ids", "token_type_ids"]:
                tensor = torch.tensor(inputs[input_name], device=self.args.device)
                tensor = tensor.view((-1, len(self.candidates)) + tensor.shape[1:])
                tensor_inputs[input_name] = tensor

            with torch.no_grad():
                mc_logits = self.model_checkpoint(**tensor_inputs)[1]

            val, ind = torch.sort(mc_logits[0], descending=True)

            ypred = self.candidates[ind[0].item()][1] # match
            tc = []
            for j in range(len(self.candidates)):
                tc.append(self.candidates[ind[j].item()][1])
            reply = {'text': ypred, 'text_candidates': tc}
        else:
            # We are in interactive of f1 evaluation mode => just sample
            with torch.no_grad():
                out_ids = sample_sequence(self.persona, self.history, self.tokenizer, self.model_checkpoint, self.args)
            out_text = self.tokenizer.decode(out_ids, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=(self.args.eval_type != 'f1'))
            reply = {'text': out_text}

        return reply

    def next_word_probability(self, partial_out):
        """Return probability distribution over next words given an input and
        partial true output. This is used to calculate the per-word perplexity.
        """
        partial_out_ids = self.tokenizer.encode(' '.join(partial_out))
        instance = build_input_from_segments(self.persona, self.history, partial_out_ids,
                                             self.tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=self.args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=self.args.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.model_checkpoint(input_ids, token_type_ids=token_type_ids)

        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        probs = F.softmax(logits[0, -1], dim=0)

        dist = {}
        for prefix_id, words in self.prefix2words.items():
            for word, ratio in words.items():
                dist[word] = probs[prefix_id].item() * ratio
        return dist

    def get_prefix2words(self, convai_dict, smoothing_freq=5):
        """ map BPE-prefix => dict(full_words beginning with BPE-prefix, associated words_counts) """
        prefix2words = defaultdict(dict)
        for i in trange(len(convai_dict)):
            word = convai_dict[i]
            freq = convai_dict.freq[word] + smoothing_freq
            bpe_tokens = self.tokenizer.bpe(word).split(' ')
            prefix_id = self.tokenizer.convert_tokens_to_ids(bpe_tokens[0])
            prefix2words[prefix_id].update(dict([(word, freq)]))

        for prefix_id, words in prefix2words.items():
            total_counts = sum(words.values())
            prefix2words[prefix_id] = dict((word, count/total_counts) for word, count in words.items())

        return prefix2words

    def share(self):
        shared = super(TransformerAgent, self).share()
        shared['tokenizer'] = self.tokenizer
        shared['model'] = self.model_checkpoint
        shared['prefix2words'] = self.prefix2words
        return shared

    def reset(self):
        self.persona = []
        self.history = []
        self.labels = []
        self.candidates = []
        self.episode_done = True
        self.observation = None

if __name__ == '__main__':
    parser = base_setup_args(None)
    parser.set_params(
        model='eval:TransformerAgent')
    opt = parser.parse_args(print_args=False)

    if opt['eval_type'] == "hits@1":
        setup_args = setup_args_hits(None)
        eval_fct = partial(eval_hits, print_parser=setup_args)
    elif opt['eval_type'] == "ppl":
        setup_args = setup_args_ppl(None)
        eval_fct = eval_ppl
    elif opt['eval_type'] == "f1":
        setup_args = setup_args_f1(None)
        eval_fct = partial(eval_f1, print_parser=setup_args)
    else:
        setup_args = setup_args_f1(None)
        eval_fct = partial(eval_f1, print_parser=setup_args)
        #raise ValueError

    setup_args.set_params(
        model='eval:TransformerAgent')
    opt = setup_args.parse_args(print_args=False)

    eval_fct(opt)