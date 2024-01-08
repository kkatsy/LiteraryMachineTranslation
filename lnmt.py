import networkx as nx
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
import itertools, copy
import torch
from matplotlib.patches import Rectangle
import seaborn as sns


def run_awesome_alignment(src, tgt, model=None, tokenizer=None):
    # model parameters
    align_layer = 8
    threshold = 1e-2

    if model is None:
        model = AutoModel.from_pretrained("aneuraz/awesome-align-with-co")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("aneuraz/awesome-align-with-co")

    # pre-processing
    sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
    token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in
                                                                             sent_tgt]
    wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for
                                                                                 x in
                                                                                 token_tgt]
    ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt',
                                                   model_max_length=tokenizer.model_max_length, truncation=True)[
                           'input_ids'], \
                       tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt',
                                                   truncation=True,
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
        tgtsrc_sub_prob = tgtsrc_sub_prob.tolist()  # torch.transpose(tgtsrc_sub_prob, 0, 1).tolist()

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

    return align_words, srctgt_prob, tgtsrc_prob


def visualize_alignment(source_sentence, tgt_sentence, alignment):
    source_len = len(source_sentence.split())
    tgt_len = len(tgt_sentence.split())
    source_nodes = ['A' + str(s1) for s1 in range(source_len)]
    gt_nodes = ['B' + str(s2) for s2 in range(tgt_len)]

    G = nx.Graph()

    edges = []
    labels_dict = {}
    nodes = []
    for s1, s2 in alignment:
        n1 = 'A' + str(s1)
        n2 = 'B' + str(s2)

        if n1 not in labels_dict:
            nodes.append(n1)
            labels_dict[n1] = source_sentence.split()[s1]

        if n2 not in labels_dict:
            nodes.append(n2)
            labels_dict[n2] = tgt_sentence.split()[s2]

        edges.append((n1, n2))

    for n in source_nodes:
        if n not in nodes:
            labels_dict[n] = source_sentence.split()[int(n[1])]

    for n in gt_nodes:
        if n not in nodes:
            labels_dict[n] = tgt_sentence.split()[int(n[1])]

    G.add_nodes_from(source_nodes, bipartite=0)  # Add the node attribute "bipartite"
    G.add_nodes_from(gt_nodes, bipartite=1)
    G.add_edges_from(edges)

    pos = dict()
    pos.update((n, (i, 2)) for i, n in enumerate(source_nodes))
    pos.update((n, (i, 1)) for i, n in enumerate(gt_nodes))

    nx.draw_networkx(G, pos=pos, with_labels=True, node_color='white', font_weight='semibold', width=2,
                     edge_color='forestgreen', labels=labels_dict)

    plt.show()


def visualize_translations_graph(translator_to_books, book_to_translator, min_translations=3, min_translators=3,
                                 title=None):
    if min_translators > 0 or min_translations > 0:

        # remove translators with less than X translations
        remove_translators = []
        for t in translator_to_books.keys():
            if len(translator_to_books[t]) < min_translations:
                books = translator_to_books[t]
                for b in books:
                    book_to_translator[b].remove(t)
                remove_translators.append(t)

        for r in remove_translators:
            translator_to_books.pop(r)

        # remove books with less than X translators
        curr_books = list(book_to_translator.keys())
        delete_books = []
        for book in curr_books:
            if len(book_to_translator[book]) < min_translators:
                book_to_translator.pop(book)
                delete_books.append(book)

        for d in delete_books:
            for t in translator_to_books.keys():
                if d in translator_to_books[t]:
                    translator_to_books[t].remove(d)

        # remove translators with less than X translations
        curr_translators = list(translator_to_books.keys())
        for t in curr_translators:
            if len(translator_to_books[t]) < min_translations:
                translator_to_books.pop(t)
                for b in book_to_translator.keys():
                    if t in book_to_translator[b]:
                        book_to_translator[b].remove(t)

        # remove books with less than X translators
        curr_books = list(book_to_translator.keys())
        for book in curr_books:
            if len(book_to_translator[book]) < min_translators:
                book_to_translator.pop(book)

    num_to_color = {1: '#B4DBFF', 2: '#A0D1FF', 3: '#8CC7FF', 4: '#78BDFF', 5: '#64B3FF', 6: '#50A9FF', 7: '#3C9FFF',
                    8: '#2895FF', 9: '#148BFF', 10: '#0081FF'}

    G = nx.Graph(book_to_translator)

    color_map = []
    node_size = []
    labels = {}
    for node in G:
        if node in book_to_translator.keys():
            color_map.append('darkorange')
            node_size.append(300)
            labels[node] = node
        else:
            num_translations = len(translator_to_books[node])
            if num_translations <= 10:
                color = num_to_color[num_translations]
            else:
                color = num_to_color[10]
            color_map.append(color)
            node_size.append(150)
            labels[node] = node.split()[-1]

    pos = nx.spring_layout(G, k=0.4, iterations=20)
    nx.draw_networkx(G, font_size='5', node_color=color_map, pos=pos, font_weight='semibold', node_size=node_size,
                     labels=labels, edge_color='gray')

    if title:
        plt.title(title)

    plt.show()

    # pruned book to translators dictionary
    return book_to_translator


def visualize_translations_heatmap(translator_to_books, book_to_translator, author_to_books, min_translations=3,
                                   min_translators=3, title=None):
    if min_translators > 0 or min_translations > 0:

        # remove translators with less than X translations
        remove_translators = []
        for t in translator_to_books.keys():
            if len(translator_to_books[t]) < min_translations:
                books = translator_to_books[t]
                for b in books:
                    book_to_translator[b].remove(t)
                remove_translators.append(t)

        for r in remove_translators:
            translator_to_books.pop(r)

        # remove books with less than X translators
        curr_books = list(book_to_translator.keys())
        delete_books = []
        for book in curr_books:
            if len(book_to_translator[book]) < min_translators:
                book_to_translator.pop(book)
                delete_books.append(book)

        for d in delete_books:
            for t in translator_to_books.keys():
                if d in translator_to_books[t]:
                    translator_to_books[t].remove(d)

        # remove translators with less than X translations
        curr_translators = list(translator_to_books.keys())
        for t in curr_translators:
            if len(translator_to_books[t]) < min_translations:
                translator_to_books.pop(t)
                for b in book_to_translator.keys():
                    if t in book_to_translator[b]:
                        book_to_translator[b].remove(t)

    # get all authors in set
    book_to_author = {}
    authors = []
    for b in book_to_translator.keys():
        for a in author_to_books.keys():
            if b in author_to_books[a]:
                book_to_author[b] = a
                if a not in authors:
                    authors.append(a)
                break

    # create matrix: translators to authors
    translation_counts = []
    translators = list(translator_to_books.keys())
    for t in translators:
        translator_to_author_counts = []
        for a in authors:
            authors_books = author_to_books[a]
            translators_books = translator_to_books[t]
            intersection = list(set(authors_books) & set(translators_books))
            num_translations_of_author = len(intersection)
            translator_to_author_counts.append(num_translations_of_author)
        translation_counts.append(translator_to_author_counts)

    f, ax = plt.subplots(figsize=(10, 6))
    ax = sns.heatmap(translation_counts, cmap="crest", xticklabels=authors, yticklabels=translators, annot=True,
                     linewidths=.5, ax=ax)
    plt.yticks(rotation=0)

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.show()

    # pruned book to translators dictionary
    return book_to_translator


def visualize_alignment_prob(probabilities, sent_tgt, sent_src, alignment, title=None):
    # heatmap of probabilities
    f, ax = plt.subplots(figsize=(9, 6))
    ax = sns.heatmap(probabilities, cmap="crest", xticklabels=sent_tgt, yticklabels=sent_src, annot=True, linewidths=.5,
                     ax=ax)
    plt.yticks(rotation=0)

    # rectangles to highlight alignment
    for align in alignment:
        x = align[0]
        y = align[1]
        ax.add_patch(Rectangle((y, x), 1, 1, fill=False, edgecolor='orangered', lw=1.7))

    if title:
        plt.title(title)

    plt.show()


def get_bleu():
    return 0


def get_bleurt():
    return 0


def bleu_vs_bleurt():
    return 0

