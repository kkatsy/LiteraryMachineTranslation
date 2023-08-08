from transformers import AutoModel, AutoTokenizer
import itertools
import torch
import pickle
import matplotlib.pyplot as plt
import copy
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

    srctgt_sub_prob = (softmax_srctgt > threshold) * softmax_srctgt
    tgtsrc_sub_prob = (softmax_tgtsrc > threshold) * softmax_tgtsrc

    srctgt_sub_prob = srctgt_sub_prob.tolist()
    tgtsrc_sub_prob = tgtsrc_sub_prob.tolist() # torch.transpose(tgtsrc_sub_prob, 0, 1).tolist()

srctgt_prob = []
tgtsrc_prob = []
for i in range(len(sent_src)):
    sub = []
    for j in range(len(sent_tgt)):
        sub.append([])
    srctgt_prob.append(copy.deepcopy(sub))
    tgtsrc_prob.append(copy.deepcopy(sub))

for i in range(len(srctgt_sub_prob)):
    for j in range(len(srctgt_sub_prob[0])):
        x = sub2word_map_src[i]
        y = sub2word_map_tgt[j]
        srctgt_prob[x][y].append(srctgt_sub_prob[i][j])
        tgtsrc_prob[x][y].append(tgtsrc_sub_prob[i][j])

get_max = True
for i in range(len(srctgt_prob)):
    for j in range(len(srctgt_prob[0])):
        probs_srctgt = srctgt_prob[i][j]
        probs_tgtsrc = tgtsrc_prob[i][j]
        if get_max:
            srctgt_prob[i][j] = max(probs_srctgt)
            tgtsrc_prob[i][j] = max(probs_tgtsrc)


# alignment
align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
align_words = set()
for i, j in align_subwords:
    align_words.add((sub2word_map_src[i], sub2word_map_tgt[j]))


als = Alignment(list(align_words))
prec = None
recall = None
aer = None

# seaborn plot

# rectangles to highlight alignment
f, ax = plt.subplots(figsize=(9, 6))
ax = sns.heatmap(srctgt_prob, cmap="crest", xticklabels=sent_tgt, yticklabels=sent_src, annot=True, linewidths=.5, ax=ax)
plt.yticks(rotation=0)

for align in align_words:
    x = align[0]
    y = align[1]
    ax.add_patch(Rectangle((y,x),1,1, fill=False, edgecolor='orangered', lw=1.7))

print(align_words)

plt.show()

f, ax = plt.subplots(figsize=(9, 6))
ax = sns.heatmap(tgtsrc_prob, cmap="crest", xticklabels=sent_tgt, yticklabels=sent_src, annot=True, linewidths=.5, ax=ax)
plt.yticks(rotation=0)

for align in align_words:
    x = align[0]
    y = align[1]
    ax.add_patch(Rectangle((y,x),1,1, fill=False, edgecolor='orangered', lw=1.7))


plt.show()

print()