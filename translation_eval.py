from nltk.translate import Alignment
from nltk.metrics import precision, recall
from nltk.translate import alignment_error_rate
from transformers import AutoModel, AutoTokenizer
import itertools
import torch
import pickle

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
src = source_translation[:126]
tgt = human_translations[0][:140]

# pre-processing
sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in
                                                                             token_tgt]
ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt',
                                               model_max_length=tokenizer.model_max_length, truncation=True)[
                       'input_ids'], \
                   tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True,
                                               model_max_length=tokenizer.model_max_length)['input_ids']
sub2word_map_src = []
for i, word_list in enumerate(token_src):
    sub2word_map_src += [i for x in word_list]
sub2word_map_tgt = []
for i, word_list in enumerate(token_tgt):
    sub2word_map_tgt += [i for x in word_list]

# alignment
align_layer = 8
threshold = 1e-3
model.eval()
with torch.no_grad():
    out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
    out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

    dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

    softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
    softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

    softmax_inter = (softmax_srctgt > threshold) * (softmax_tgtsrc > threshold)


# alignment
align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
align_words = set()
for i, j in align_subwords:
    align_words.add((sub2word_map_src[i], sub2word_map_tgt[j]))


als = Alignment(list(align_words))
prec = None # precision(Alignment([(0,0), (1,1), (2,2), (3,3), (1,2), (2,1)]), als.alignment)
recall = None # recall(Alignment([(0,0), (1,1), (2,2), (3,3), (1,2), (2,1)]), als.alignment)
aer = None # alignment_error_rate(als.alignment, my_als.alignment)