from transformers import AutoModel, AutoTokenizer
import itertools
import torch
import pickle
import matplotlib.pyplot as plt
import lnmt
from matplotlib.patches import Rectangle
import seaborn as sns
sns.set()

# load model
model = AutoModel.from_pretrained("aneuraz/awesome-align-with-co")
tokenizer = AutoTokenizer.from_pretrained("aneuraz/awesome-align-with-co")

# model parameters
align_layer = 8
threshold = 1e-2

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
src = source_translation[:36]
tgt = human_translations[0][:26]
# src = source_translation[:126]
# tgt = human_translations[0][:140]

sent_src, sent_tgt = src.strip().split(), tgt.strip().split()

align_words, srctgt_prob, tgtsrc_prob = lnmt.run_awesome_alignment(src, tgt)

# seaborn plots
lnmt.visualize_alignment_prob(srctgt_prob, sent_tgt, sent_src, align_words, title="Source to Target Probabilities")

lnmt.visualize_alignment_prob(tgtsrc_prob, sent_tgt, sent_src, align_words, title="Target to Source Probabilities")
