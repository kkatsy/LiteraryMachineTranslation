import pickle
import networkx as nx
import matplotlib.pyplot as plt
from nltk.translate import Alignment
import lnmt


par3_fp = "par3_top.pickle"
par3 = pickle.load(open(par3_fp, 'rb'))


for i, par in enumerate(par3['ru']['we']['source_paras']):
    if 'Проснулся: умеренный' in par:
        break

source_translation = par3['ru']['we']['source_paras'][i]

gt_translation = par3['ru']['we']['gt_paras'][631]

human_translations = []
for t in par3['ru']['we']['translator_data'].keys():
    human_translations.append(par3['ru']['we']['translator_data'][t]['translator_paras'][631])

# define inputs
# src = source_translation[:36]
# tgt = human_translations[0][:26]
src = source_translation[:126]
tgt = human_translations[0][:140]


align_words, srctgt_prob, tgtsrc_prob = lnmt.run_awesome_alignment(src, tgt)

sent_src, sent_tgt = src.strip().split(), tgt.strip().split()

print(align_words)

lnmt.visualize_alignment(sent_src, sent_tgt, align_words)


