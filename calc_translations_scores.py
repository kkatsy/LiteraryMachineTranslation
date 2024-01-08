import tensorflow as tf
import time
from bleurt import score
import pickle
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sacrebleu import corpus_bleu, BLEU
from nltk.tokenize import RegexpTokenizer
from statistics import mean, stdev
from tqdm import tqdm

BOOK = 'the_idiot'

par3_fp = "par3_top.pickle"
par3 = pickle.load(open(par3_fp, 'rb'))

# pre-process
book_paras = par3['ru'][BOOK]

include_idx = []
for i, par in enumerate(book_paras['source_paras']):
    if len(par.split()) > 4:
        include_idx.append(i)
print('Original num paras: ', len(book_paras['source_paras']))
print('>4 words num paras: ', len(include_idx))

# for each instance: get corpus BLEU of machine + human translations
index_to_score = {} # index, sent_num to average bleu
index_to_translations = {}
dropped = 0
for i in tqdm(include_idx):

    gt_translation = book_paras['gt_paras'][i]
    human_translations = {}
    lengths = []
    for j, translator in enumerate(book_paras['translator_data'].keys()):
        translation = book_paras['translator_data'][translator]['translator_paras'][i]
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


with open('book_scores/' + BOOK + '_scores.pickle', 'wb') as handle:
    pickle.dump(index_to_score, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('book_scores/' + BOOK + '_translations.pickle', 'wb') as handle:
    pickle.dump(index_to_translations, handle, protocol=pickle.HIGHEST_PROTOCOL)

