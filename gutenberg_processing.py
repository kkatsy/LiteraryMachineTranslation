import re

with open("russian_lit/ru_books_text_cleaner/PoorFolk_Hogarth.txt", "r") as fp:
    book_text = fp.read()

# remove footnotes: [1] ...
# replace _word_ with word (can just remove underscores)
# replace newline with space
# remove [#] from text

book_text = book_text.replace('_', '')

book_text = re.sub('(.)\n(?!\n)', r'\1 ', book_text)

book_text = re.sub(r'^\[.*\].*\n?', '', book_text)

book_text = re.sub(r'\[.*?\]', '', book_text)

with open("russian_lit/ru_books_text_cleaner/PoorFolk_Hogarth.txt", "w") as text_file:
    text_file.write(book_text)