# replace Il -> I'll
# 'T or "T -> 'I
# Iam -> I am, Ihave -> I have
# nums right after dot or word (no space)

# lowercase -> remove newline connect to paragraph
# -\n -> connect

import glob
import re

# from spellchecker import SpellChecker
#
# spell = SpellChecker()

path = 'russian_lit/ru_books_text_raw'
save_path = 'russian_lit/ru_books_text_cleaner'
files = glob.glob(path + '/*.txt')
for f in files:
    book_title = (f.split('/')[-1]).split('.')[0]
    with open(f, "r") as fp:
        book_text = fp.read()

    # remove word splits
    book_text = book_text.replace('-\n', '').replace('-\n\n', '')
    book_text = re.sub(r"\r?\n(?=\s?[a-z].*[?.!,;\-'\"])", "", book_text)

    # remove footnotes
    # book_text = re.sub(r"\r?\n(?=\s?[a-z].*[?.!,;\-'\"])", " ", book_text) -> string with sole number at end

    # quotes
    book_text = book_text.replace('“', '\"').replace('’', '\'').replace('”', '\"').replace('‘', '\'')

    # common misspellings
    book_text = book_text.replace('Il ', 'I\'ll').replace('Iam', "I am").replace('Ishould', 'I should')
    book_text = book_text.replace('Ijust', 'I just').replace('Ihave', 'I have').replace('Ican\'t', 'I can\'t')
    book_text = book_text.replace('Tshould', 'I should').replace('Tjust', 'I just').replace('Thave', "I have")
    book_text = book_text.replace('Ihad', 'I had').replace('I\'ma', 'I\'m a').replace('T\'ve', 'I\'ve')
    book_text = book_text.replace('Tl ', 'I\'ll').replace("\"Tm", "I\'m").replace('Tam', "I am")
    book_text = book_text.replace('\'T ', '\'I ').replace('\"T ', '\"I ')
    book_text = book_text.replace(' T ', ' I ').replace('T\'ll', 'I\'ll').replace("Tt", "It")
    book_text = book_text.replace('\'J ', '\'I ').replace('\"J ', '\"I ').replace(' J ', ' I ')
    book_text = book_text.replace('\'7 ', '\'I ').replace('\"7 ', '\"I ')
    book_text = book_text.replace('\'1 ', '\'I ').replace('\"1 ', '\"I ')
    book_text = book_text.replace('witha', 'with a').replace('hada', 'had a').replace('hima', 'him a').replace('froma', 'from a')
    book_text = book_text.replace('hey\'I\'ll ', 'hey\'ll ').replace('Icould', 'I could')
    book_text = book_text.replace('|', 'I').replace('Iknew', 'I knew').replace('amoment', 'a moment').replace('Icame', 'I came')

    book_text = re.sub("(I\'ll)(?=\w)", 'I\'ll\2', book_text)
    book_text = book_text.replace('', ' ')

    book_text = re.sub("( ing )", 'ing ', book_text)

    with open(save_path + "/" + book_title + ".txt", "w") as text_file:
        text_file.write(book_text)
