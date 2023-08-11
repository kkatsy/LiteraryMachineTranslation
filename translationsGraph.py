import re
import networkx as nx
import matplotlib.pyplot as plt
import copy

translators_per_book = 4
translations_per_translator = 3

with open("russian_translations.txt", "r") as f:
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
remove_translators = []
for t in translator_to_books.keys():
    if len(translator_to_books[t]) < translations_per_translator:
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
    if len(book_to_translator[book]) < translators_per_book:
        book_to_translator.pop(book)
        delete_books.append(book)

for d in delete_books:
    for t in translator_to_books.keys():
        if d in translator_to_books[t]:
            translator_to_books[t].remove(d)

# remove translators with less than X translations
curr_translators = list(translator_to_books.keys())
for t in curr_translators:
    if len(translator_to_books[t]) < translations_per_translator:
        translator_to_books.pop(t)
        for b in book_to_translator.keys():
            if t in book_to_translator[b]:
                book_to_translator[b].remove(t)


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

plot_title = str(translators_per_book) + " or more translators per book, " + str(translations_per_translator) + " or more translations per translator"
plt.title(plot_title)
plt.show()

# Print
print(book_to_translator)
print('Num books: ', len(book_to_translator.keys()))
print('Num translators: ', len(translator_to_books.keys()))
print('Translators: ', list(set(translator_to_books.keys())))
print('Books: ', list(set(book_to_translator.keys())))

print("\n\n")

count = 1
for b in sorted(book_to_translator.keys()):
    for tr in book_to_translator[b]:
        print(count, ') ', b, ' by Fyodor Dostoevsky, translated by ', tr)
        count+=1
