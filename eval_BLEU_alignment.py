import pickle
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from statistics import mean
from lnmt import visualize_alignment, run_awesome_alignment

# take 'all src pars from We
par3_fp = "par3_top.pickle"
par3 = pickle.load(open(par3_fp, 'rb'))

# pre-process
we_paras = par3['ru']['we']

include_idx = []
for i,par in enumerate(we_paras['source_paras']):
    if len(par.split()) > 4:
        include_idx.append(i)
print('Original num paras: ', len(we_paras['source_paras']))
print('>4 words num paras: ', len(include_idx))

# for each instance: get corpus BLEU of machine + human translations
index_to_bleu = {} # index to average bleu
for i in include_idx:
    gt_translation = we_paras['gt_paras'][i]
    human_translations = []
    for translator in we_paras['translator_data'].keys():
        translation = we_paras['translator_data'][translator]['translator_paras'][i]
        human_translations.append(translation)

    bleus = []
    for h in human_translations:
        reference = gt_translation
        hypothesis = h
        bleus.append(corpus_bleu([reference], [hypothesis], weights = [1]))

    average_bleu = mean(bleus)
    index_to_bleu[i] = average_bleu

max_bleu_idx = max(index_to_bleu, key=index_to_bleu.get)
min_bleu_idx = min(index_to_bleu, key=index_to_bleu.get)
print('min bleu: ', index_to_bleu[min_bleu_idx], ' , min index: ', min_bleu_idx )
print('min bleu length: ', len(we_paras['gt_paras'][min_bleu_idx]))

print('max bleu: ', index_to_bleu[max_bleu_idx], ' , max index: ', max_bleu_idx )
print('max bleu length: ', len(we_paras['gt_paras'][max_bleu_idx]))
# bleus clues: identifying non-compositional language via bleu and word alignment

top_3_bleu = sorted(index_to_bleu, key=index_to_bleu.get, reverse=True)[:3]
bottom_3_bleu = sorted(index_to_bleu, key=index_to_bleu.get, reverse=False)[:3]


# for each instance: get sentence BLEU of machine + human translations

# for each instance: get BLEURT of machine + human translations

# look at word alignments of best + worst cases
worst_src = None
worst_tgt = None
align_words_worst, srctgt_prob_worst, tgtsrc_prob_worst = run_awesome_alignment(worst_src, worst_tgt)
visualize_alignment(align_words_worst, worst_src, worst_tgt)

best_src = None
best_tgt = None
align_words_best, srctgt_prob_best, tgtsrc_prob_best = run_awesome_alignment(best_src, best_tgt)
visualize_alignment(align_words_best, best_src, best_tgt)