import re
from lnmt import visualize_translations_graph, visualize_translations_heatmap

EXCLUDE_TRANSLATOR = ["David Magarshack"]

with open("ru_books_dataset.txt", "r") as f:
    translations_list = f.read().split("\n\n")

translator_to_books = {}
book_to_translator = {}
author_to_translators = {}
author_to_book = {}
for book_info in translations_list:
    info = book_info.split('\n')

    title, author = info[0].split(', ')

    author_to_translators[author] = []
    book_to_translator[title] = []

    if author not in author_to_book:
        author_to_book[author] = []
    author_to_book[author].append(title)

    translators_list = info[1:]

    for a in translators_list:
        translator = re.search(r"\)(.*) -", a).group(1).strip()

        if translator not in EXCLUDE_TRANSLATOR:
            author_to_translators[author].append(translator)

            book_to_translator[title].append(translator)

            if translator not in translator_to_books.keys():
                translator_to_books[translator] = []
            translator_to_books[translator].append(title)
        else:
            print()

# visualization params
min_translators_per_book = 0
min_translations_per_translator = 0
plot_title = str(min_translators_per_book) + " or more translators per book, " + str(
    min_translations_per_translator) + " or more translations per translator"

# get graph of translators to books
pruned_book_to_translator = visualize_translations_graph(translator_to_books, book_to_translator,
                                                         min_translations_per_translator, min_translators_per_book,
                                                         title=plot_title)

# get heatmap of translators to authors
pruned_book_to_translator = visualize_translations_heatmap(translator_to_books, book_to_translator, author_to_book,
                                                           min_translations_per_translator, min_translators_per_book,
                                                           title=plot_title)
