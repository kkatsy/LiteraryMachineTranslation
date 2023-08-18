import tensorflow as tf
import time
from bleurt import score
import pickle
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

references = tf.constant(["This is a test.", "This is also the test!", "That is a test."])
candidates = tf.constant(["This is the test."])

start_time = time.time()

bleurt_ops = score.create_bleurt_ops()
bleurt_out = bleurt_ops(references=references, candidates=candidates)

end_time = time.time()

total_time = end_time - start_time

assert bleurt_out["predictions"].shape == (1,)
print(float(bleurt_out["predictions"][0]))
print("Seconds: ", total_time)


######################################################################################

# take 'all src pars from We
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
average_len = 0
for i in include_idx:
    gt_translation = we_paras['gt_paras'][i]
    human_translations = []
    for translator in we_paras['translator_data'].keys():
        translation = we_paras['translator_data'][translator]['translator_paras'][i]
        human_translations.append(translation)

    references = [h for h in human_translations]
    hypothesis = [gt_translation]
    bleurt_ops = score.create_bleurt_ops()
    bleurt_out = bleurt_ops(references=tf.constant(references), candidates=tf.constant(hypothesis))
    bleu = float(bleurt_out["predictions"][0])

    # bleus = []
    # for r in references:
    #     bleurt_out = bleurt_ops(references=tf.constant([r]), candidates=tf.constant(hypothesis))
    #     a_bleu = float(bleurt_out["predictions"][0])
    #     bleus.append(a_bleu)

    index_to_bleu[i] = bleu
    index_to_translations[i] = {'gt': gt_translation, 'hum': human_translations}


bottom_bleu = sorted(index_to_bleu, key=index_to_bleu.get, reverse=False)[:10]
top_bleu = sorted(index_to_bleu, key=index_to_bleu.get, reverse=True)[:10]

print('BOTTOM')
for b in bottom_bleu:# + bottom_bleu:
    print('BLEURT: ', index_to_bleu[b])
    print('Sentence: ', we_paras['source_paras'][b])
    print('Translations: ', index_to_translations[b])
    print()

print('TOP')
for b in top_bleu:# + bottom_bleu:
    print('BLEURT: ', index_to_bleu[b])
    print('Sentence: ', we_paras['source_paras'][b])
    print('Translations: ', index_to_translations[b])
    print()