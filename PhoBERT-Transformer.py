import torch
import argparse

from transformers import RobertaConfig
from transformers import RobertaModel

from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

# Load model
config = RobertaConfig.from_pretrained(
    "/Users/trinhgiang/Downloads/PhoBERT_base_transformers/config.json"
)
phobert = RobertaModel.from_pretrained(
    "/Users/trinhgiang/Downloads/PhoBERT_base_transformers/model.bin",
    config=config
)

# Load BPE encoder
parser = argparse.ArgumentParser()
parser.add_argument('--bpe-codes',
    default="/Users/trinhgiang/Downloads/PhoBERT_base_transformers/bpe.codes",
    required=False,
    type=str,
    help='path to fastBPE BPE'
)
args = parser.parse_args()
bpe = fastBPE(args)

# INPUT TEXT IS WORD-SEGMENTED!
line = "Tôi là sinh_viên trường đại_học Công_nghệ ."

# Load the dictionary
vocab = Dictionary()
vocab.add_from_file("/Users/trinhgiang/Downloads/PhoBERT_base_transformers/dict.txt")

# Encode the line using fast BPE & Add prefix <s> and suffix </s>
subwords = '<s> ' + bpe.encode(line) + ' </s>'

# Map subword tokens to corresponding indices in the dictionary
input_ids = vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()

# Convert into torch tensor
all_input_ids = torch.tensor([input_ids], dtype=torch.long)

# Extract features
features = phobert(all_input_ids)

print('a')