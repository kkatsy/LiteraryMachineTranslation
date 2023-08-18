import pickle
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from sacrebleu.metrics import BLEU, CHRF, TER
import nltk
from statistics import mean
import spacy
from nltk.tokenize import RegexpTokenizer

# from lnmt import visualize_alignment, run_awesome_alignment
import re

nlp = spacy.load ("en_core_web_sm-3.6.0/en_core_web_sm/en_core_web_sm-3.6.0/")
# take 'all src pars from We
par3_fp = "par3_top.pickle"
par3 = pickle.load(open(par3_fp, 'rb'))

# pre-process
we_paras = par3['ru']['we']

include_idx = []
for i, par in enumerate(we_paras['source_paras']):
    if len(par.split()) > 4:
        include_idx.append(i)
print('Original num paras: ', len(we_paras['source_paras']))
print('>4 words num paras: ', len(include_idx))

# for each instance: get corpus BLEU of machine + human translations

index_to_bleu = {} # index, sent_num to average bleu
index_to_translations = {}
tokenizer = RegexpTokenizer(r'\w+')
cc = SmoothingFunction()
average_len = 0
for i in include_idx:
    gt_translation = we_paras['gt_paras'][i]
    human_translations = []
    for translator in we_paras['translator_data'].keys():
        translation = we_paras['translator_data'][translator]['translator_paras'][i]
        human_translations.append(translation)

    reference = [tokenizer.tokenize(h) for h in human_translations]
    hypothesis = tokenizer.tokenize(gt_translation)
    average_bleu = corpus_bleu([reference], [hypothesis], smoothing_function=cc.method4)

    for r in reference:
        average_len += len(r)/(len(reference)*len(include_idx))

    index_to_bleu[i] = average_bleu
    index_to_translations[i] = {'gt': gt_translation, 'hum': human_translations}


bottom_bleu = sorted(index_to_bleu, key=index_to_bleu.get, reverse=False)[:5]
top_bleu = sorted(index_to_bleu, key=index_to_bleu.get, reverse=True)[:5]


index_to_norm_bleu = {}
for k,v in index_to_bleu.items():
    translation_len = 0
    for translator in we_paras['translator_data'].keys():
        translation = we_paras['translator_data'][translator]['translator_paras'][k]
        translation_len += len(tokenizer.tokenize(translation))
    translation_len /= len(list(we_paras['translator_data'].keys()))
    norm_v = v * (translation_len/average_len)
    index_to_norm_bleu[k] = norm_v

min_bleu = min(index_to_norm_bleu.values())
max_bleu = max(index_to_norm_bleu.values())
index_to_normalized = {}
for k,v in index_to_norm_bleu.items():
    scaled_v = (v - min_bleu)/(max_bleu - min_bleu)
    index_to_normalized[k] = scaled_v

bottom_norm_bleu = sorted(index_to_normalized, key=index_to_normalized.get, reverse=False)[:10]
top_norm_bleu = sorted(index_to_normalized, key=index_to_normalized.get, reverse=True)[:10]

print('BOTTOM')
for b in bottom_bleu:# + bottom_bleu:
    print('BLEU: ', index_to_bleu[b])
    # print('Normalized BLEU: ', index_to_normalized[b])
    print('Sentence: ', we_paras['source_paras'][b])
    print('Translations: ', index_to_translations[b])
    print()

print('TOP')
for b in top_bleu: # + bottom_norm_bleu:
    print('BLEU: ', index_to_bleu[b])
    # print('Normalized BLEU: ', index_to_normalized[b])
    print('Sentence: ', we_paras['source_paras'][b])
    print('Translations: ', index_to_translations[b])
    print()
# for each instance: get sentence BLEU of machine + human translations

