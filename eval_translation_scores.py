import pickle
import glob, os
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

scores_files = glob.glob('book_scores/the_idiot_scores.pickle')

with open(scores_files[0], 'rb') as handle:
    scores = pickle.load(handle)

ave_bleu, comb_bleu = [], []
ave_bleurt, comb_bleurt = [], []
for score_dict in scores.values():
    comb_bleu.append(score_dict['comp_bleu']/100)
    comb_bleurt.append(score_dict['comb_bleurt'])
    ave_bleu.append(score_dict['ave_blue']/100)
    ave_bleurt.append(score_dict['ave_bleurt'])

comb_corr, _ = pearsonr(comb_bleu, comb_bleurt)
ave_corr, _ = pearsonr(ave_bleu, ave_bleurt)
bleurt_corr, _ = pearsonr(ave_bleurt, comb_bleurt)
bleu_corr, _ = pearsonr(ave_bleu, comb_bleu)

print('Combined BLEU vs BLEURT correlation: ', comb_corr)
print('Average BLEU vs BLEURT correlation: ', ave_corr)
print('BLEU average vs combined correlation: ', bleu_corr)
print('BLEURT average vs combined correlation: ', bleurt_corr)


# correlation plot

figure_1, axis = plt.subplots(2, 2)

axis[0, 0].set_title("Combined BLEU vs BLEURT correlation")
axis[0, 0].set_xlabel("BLEU")
axis[0, 0].set_ylabel("BLEURT")
axis[0, 0].scatter(comb_bleu, comb_bleurt)

axis[0, 1].set_title("Average BLEU vs BLEURT correlation")
axis[0, 1].set_xlabel("BLEU")
axis[0, 1].set_ylabel("BLEURT")
axis[0, 1].scatter(ave_bleu, ave_bleurt)

axis[1, 0].set_title("BLEU average vs combined correlation")
axis[1, 0].set_xlabel("average")
axis[1, 0].set_ylabel("combined")
axis[1, 0].scatter(ave_bleu, comb_bleu)

axis[1, 1].set_title("BLEURT average vs combined correlation")
axis[1, 1].set_xlabel("average")
axis[1, 1].set_ylabel("combined")
axis[1, 1].scatter(ave_bleurt, comb_bleurt)

plt.tight_layout()
plt.show()


# histogram

figure_2, axis = plt.subplots(2, 2)

axis[0, 0].set_title("Average BLEU distribution")
axis[0, 0].set_xlabel("BLEU")
axis[0, 0].set_ylabel("count")
axis[0, 0].hist(ave_bleu, edgecolor="red", bins=20)

axis[0, 1].set_title("Average BLEURT distribution")
axis[0, 1].set_xlabel("BLEURT")
axis[0, 1].set_ylabel("count")
axis[0, 1].hist(ave_bleurt, edgecolor="red", bins=20)

axis[1, 0].set_title("Combined BLEU distribution")
axis[1, 0].set_xlabel("BLEU")
axis[1, 0].set_ylabel("count")
axis[1, 0].hist(comb_bleu, edgecolor="red", bins=20)

axis[1, 1].set_title("Combined BLEURT distribution")
axis[1, 1].set_xlabel("BLEURT")
axis[1, 1].set_ylabel("count")
axis[1, 1].hist(comb_bleurt, edgecolor="red", bins=20)

plt.tight_layout()
plt.show()