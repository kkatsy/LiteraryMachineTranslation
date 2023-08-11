from nltk.translate import Alignment
from nltk.metrics import precision, recall
from nltk.translate import alignment_error_rate
import itertools
import pickle
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
src = source_translation[:126]
tgt = human_translations[0][:140]

align_words, srctgt_prob, tgtsrc_prob = lnmt.run_awesome_alignment(src, tgt)

als = Alignment(list(align_words))
prec = None # precision(Alignment([(0,0), (1,1), (2,2), (3,3), (1,2), (2,1)]), als.alignment)
recall = None # recall(Alignment([(0,0), (1,1), (2,2), (3,3), (1,2), (2,1)]), als.alignment)
aer = None # alignment_error_rate(als.alignment, my_als.alignment)