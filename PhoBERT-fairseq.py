import tensorflow as tf
import json
import numpy as np
# Load PhoBERT-base in fairseq
from fairseq.models.roberta import RobertaModel
# Load rdrsegmenter from VnCoreNLP
from vncorenlp import VnCoreNLP

rdrsegmenter = VnCoreNLP("/Users/trinhgiang/Downloads/VnCoreNLP-master/VnCoreNLP-1.1.1.jar", annotators="wseg",
                         max_heap_size='-Xmx500m')

phobert = RobertaModel.from_pretrained('/Users/trinhgiang/Downloads/PhoBERT_base_fairseq', checkpoint_file='model.pt')
phobert.eval()  # disable dropout (or leave in train mode to finetune)

# Incorporate the BPE encoder into PhoBERT-base
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq import options

parser = options.get_preprocessing_parser()
parser.add_argument('--bpe-codes', type=str, help='path to fastBPE BPE',
                    default="/Users/trinhgiang/Downloads/PhoBERT_base_fairseq/bpe.codes")
args = parser.parse_args()
phobert.bpe = fastBPE(args)  # Incorporate the BPE encoder into PhoBERT


def array_to_sentence(array):
    sentence = ''
    for word in array:
        sentence += ' '
        sentence += word
    return sentence.strip().replace("_", " ")


def sentence_analysis(sentence: str):
    result = {}
    tokens = []
    sentence_loss = 0.0
    tokenized_sentence = rdrsegmenter.tokenize(sentence.strip())[0]
    # tokenized_sentence = sentence.strip().split(' ')
    for i in range(len(tokenized_sentence)):
        word_check = tokenized_sentence[i]
        sentence_masked = tokenized_sentence.copy()
        sentence_masked[i] = '<mask>'
        sentence_check = array_to_sentence(sentence_masked)
        word_loss = phobert.check_mask(sentence_check, word_check)
        sentence_loss += word_loss
        tokens.append({"token": word_check,
                       "prob": float(np.exp(-word_loss))})
    word_count_per_sent = len(tokenized_sentence)
    result["tokens"] = tokens
    result["ppl"] = float(np.exp(sentence_loss / word_count_per_sent))
    return result


def sentence_analysis_2(sentence: str):
    result = {}
    tokens = []
    sentence_loss = 0.0
    tokenized_sentence = rdrsegmenter.tokenize(sentence.strip())[0]
    # tokenized_sentence = sentence.strip().split(' ')
    for i in range(len(tokenized_sentence)):
        word_check = tokenized_sentence[i]
        sentence_masked = tokenized_sentence.copy()
        sentence_masked[i] = '<mask>'
        # ngram = 0
        # if ((i+ngram) < len(tokenized_sentence)):
        #     sentence_check = array_to_sentence(sentence_masked[0:i+ngram+1])
        if (False):
            sentence_check = array_to_sentence(sentence_masked[0:i+1])
        else:
            sentence_check = array_to_sentence(sentence_masked)
        word_loss = phobert.check_mask(sentence_check, word_check)
        sentence_loss += word_loss
        tokens.append({"token": word_check,
                       "prob": float(np.exp(-word_loss))})
    word_count_per_sent = len(tokenized_sentence)
    result["tokens"] = tokens
    result["ppl"] = float(np.exp(sentence_loss / word_count_per_sent))
    return result


def find_break_sentence(sentence, threshold):
    result_prob_token = sentence_analysis(sentence)
    if (result_prob_token["ppl"] < 20):
        return 0
    else:
        i = 0
        for token in result_prob_token["tokens"]:
            if (i != 0 and token['prob'] < threshold):
                return i
            i += 1
        return 0


def split_sentence(sentence, threshold):
    sentences = []
    sentence_check = sentence
    while (find_break_sentence(sentence_check, threshold) != 0):
        tokenized_sentence_check = rdrsegmenter.tokenize(sentence_check.strip())[0]
        idx_break = find_break_sentence(sentence_check, threshold)
        sentence_tmp_arr = tokenized_sentence_check[:idx_break]
        sentence_tmp_txt = array_to_sentence(sentence_tmp_arr)
        sentences.append(sentence_tmp_txt)
        sentence_tmp_arr = tokenized_sentence_check[idx_break:]
        sentence_tmp_txt = array_to_sentence(sentence_tmp_arr)
        sentence_check = sentence_tmp_txt
    sentences.append(sentence_check)
    return sentences


#
sentence = 'Ngày sinh: Giới tính: Điện thoại: Email: Địa chỉ:'
tokenized_sentence = rdrsegmenter.tokenize(sentence.strip())[0]
result = sentence_analysis_2(sentence)
with tf.gfile.GFile("/Users/trinhgiang/Downloads/bert-as-language-model-master/tmp/lm_output_20000/output_PhoBERT_2.json",
                    "w") as writer:
    writer.write(json.dumps(result, indent=2, ensure_ascii=False))

# print(split_sentence(sentence, 0.0001))
