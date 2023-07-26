import re
import networkx as nx
import matplotlib.pyplot as plt

with open("translations.txt", "r") as f:
    translations_list = f.read().split("\n\n")

translator_to_books = {}
book_to_translator = {}
for book_info in translations_list:
    info = book_info.split('\n')

    title = info[0]
    book_to_translator[title] = []

    authors_list = info[1:]
    for a in authors_list:
        translator = re.search(r"\)(.*) -", a).group(1)

        book_to_translator[title].append(translator)

        if translator not in translator_to_books.keys():
            translator_to_books[translator] = []
        translator_to_books[translator].append(title)

# remove translators with only one translation
translator_to_num = {}
for t in translator_to_books.keys():
    if len(translator_to_books[t]) < 3:
        books = translator_to_books[t]
        for b in books:
            book_to_translator[b].remove(t)
    else:
        translator_to_num[t] = len(translator_to_books[t])

print(set(translator_to_num.values()))
# R = 0
# G = 129
# for i in range(1,10):
#     R += 20
#     G += 10
#     print(i, ": R = ", R, ", G = ", G)

num_to_color = {1: '#B4DBFF', 2: '#A0D1FF', 3: '#8CC7FF', 4: '#78BDFF', 5: '#64B3FF', 6: '#50A9FF', 7: '#3C9FFF',
                8: '#2895FF', 9: '#148BFF', 10: '#0081FF'}

curr_books = list(book_to_translator.keys())
for book in curr_books:
    if len(book_to_translator[book]) < 3:
        book_to_translator.pop(book)

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
        color = num_to_color[num_translations]
        color_map.append(color)
        node_size.append(150)
        labels[node] = node.split()[-1]

pos = nx.spring_layout(G, k=0.4, iterations=20)
nx.draw_networkx(G, font_size='5', node_color=color_map, pos=pos, font_weight='semibold', node_size=node_size,
                 labels=labels, edge_color='gray')
plt.show()

print()
