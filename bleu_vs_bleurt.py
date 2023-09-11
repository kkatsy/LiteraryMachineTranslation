import tensorflow as tf
import time
from bleurt import score
import pickle
from scipy.stats import pearsonr
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sacrebleu import corpus_bleu, BLEU
from nltk.tokenize import RegexpTokenizer
from statistics import mean, stdev


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
index_to_score = {} # index, sent_num to average bleu
index_to_translations = {}
dropped = 0
for i in include_idx:
    gt_translation = we_paras['gt_paras'][i]
    human_translations = {}
    lengths = []
    for j, translator in enumerate(we_paras['translator_data'].keys()):
        translation = we_paras['translator_data'][translator]['translator_paras'][i]
        human_translations['hum'+str(j+1)] = translation
        lengths.append(len(translation))

    ##############################################
    # mean of separate sacrebleu scores
    bleu_list = []
    hyp = [gt_translation]
    for h in human_translations.values():
        ref = [[h]]
        bleu = corpus_bleu(hyp, ref, lowercase=True)
        bleu_list.append(bleu.score)

    mean_bleu = mean(bleu_list)

    ##############################################
    # single sacrebleu score for all
    ref = [[h] for h in human_translations.values()]
    bleu = corpus_bleu(hyp, ref, lowercase=True).score

    ##############################################
    # bleurt for all
    references = [h for h in human_translations.values()]
    hypothesis = [gt_translation]
    bleurt_ops = score.create_bleurt_ops()
    bleurt_out = bleurt_ops(references=tf.constant(references), candidates=tf.constant(hypothesis))
    bleurt = float(bleurt_out["predictions"][0])

    ##############################################
    # mean of separate bleurt scores
    bleurt_list = []
    hyp = [gt_translation]
    for h in human_translations.values():
        ref = [h]
        bleurt_out = bleurt_ops(references=tf.constant(ref), candidates=tf.constant(hyp))
        bleurt_list.append(float(bleurt_out["predictions"][0]))

    mean_bleurt = mean(bleurt_list)

    ##############################################
    score_dict = {'comp_bleu': bleu, 'ave_blue': mean_bleu, 'comb_bleurt': bleurt, 'ave_bleurt': mean_bleurt}
    index_to_score[i] = score_dict
    gt = {'gt': gt_translation}
    index_to_translations[i] = {**gt, **human_translations}

# bottom_bleu = sorted(index_to_bleu, key=index_to_bleu.get, reverse=False)[:10]
# top_bleu = sorted(index_to_bleu, key=index_to_bleu.get, reverse=True)[:10]