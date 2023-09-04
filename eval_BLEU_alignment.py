import pickle
from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu
from sacrebleu import corpus_bleu, BLEU
from nltk.tokenize import RegexpTokenizer
from statistics import mean, stdev

# refs = [['The dog bit the man.']]
# sys = ['The dog had bit the man.']
#
# bleu_sacre = corpus_bleu(sys, refs, lowercase=True)
# tokenizer = RegexpTokenizer(r'\w+')
# ref = tokenizer.tokenize('The dog bit the man.')
# hyp = tokenizer.tokenize('The dog had bit the man.')
# bleu_nltk = nltk_corpus_bleu([hyp], [ref])
#

#############################################################################


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
dropped = 0
for i in include_idx:
    gt_translation = we_paras['gt_paras'][i]
    human_translations = {}
    lengths = []
    for j, translator in enumerate(we_paras['translator_data'].keys()):
        translation = we_paras['translator_data'][translator]['translator_paras'][i]
        human_translations['hum'+str(j+1)] = translation
        lengths.append(len(translation))

    len_mean = mean(lengths)
    len_sd = stdev(lengths)
    filtrd_human_translations = {}
    for k, v in human_translations.items():
        this_len = len(v)
        if this_len > len_mean + 1.75*len_sd:
            dropped += 1
            continue
        if this_len < len_mean - 1.75*len_sd:
            dropped += 1
            continue
        filtrd_human_translations[k] = v

    bleu_list = []
    hyp = [gt_translation]
    for h in filtrd_human_translations.values():
        ref = [[h]]
        bleu = corpus_bleu(hyp, ref, lowercase=True)
        bleu_list.append(bleu.score)

    ref = [[h] for h in filtrd_human_translations.values()]
    sacre_bleu = corpus_bleu(hyp, ref, lowercase=True)
    mean_bleu = mean(bleu_list)

    index_to_bleu[i] = sacre_bleu.score
    gt = {'gt': gt_translation}
    index_to_translations[i] = {**gt, **filtrd_human_translations}

bottom_bleu = sorted(index_to_bleu, key=index_to_bleu.get, reverse=False)[:10]
top_bleu = sorted(index_to_bleu, key=index_to_bleu.get, reverse=True)[:10]

print('dropped: ', dropped)

print('BOTTOM')
for b in bottom_bleu:# + bottom_bleu:
    print('Index: ', b)
    print('BLEU: ', index_to_bleu[b])
    print('Sentence: ', we_paras['source_paras'][b])
    print('Translations:')
    for k, v in index_to_translations[b].items():
        print(k, ': ', v)
    print()

print('TOP')
for b in top_bleu: # + bottom_norm_bleu:
    print('Index: ', b)
    print('BLEU: ', index_to_bleu[b])
    print('Sentence: ', we_paras['source_paras'][b])
    print('Translations:')
    for k, v in index_to_translations[b].items():
        print(k, ': ', v)
    print()

