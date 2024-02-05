
# with open("russian_lit/ru_books_text_cleaner/non_copyrighted/AnnaKarenina_Garnett.txt", "r") as fp:
#     book_text = fp.read()
#
# # book_text = book_text.replace("⁠", "").replace(" ", "")
#
# # split into chapters by roman numerals
# # remove line after #, remove book # -> Brothers Karamazov
# # split by date (i.e. April XX) instead of chapters -> Poor Folk
# book_text = book_text.replace('“', '\"').replace('’', '\'').replace('”', '\"')
# with open("russian_lit/ru_books_text_cleaner/non_copyrighted/AnnaKarenina_Garnett.txt", "w") as text_file:
#     text_file.write(book_text)

roman_dict = {}
import glob
path = 'russian_lit/ru_books_text_cleaner/non_copyrighted'
save_path = 'russian_lit/ru_books_text_cleaner/non_copyrighted'
files = glob.glob(path + '/*.txt')
for f in files:
    book_title = (f.split('/')[-1]).split('.')[0]
    with open(f, "r") as fp:
        book_text = fp.read()

    # book_text = book_text.replace("⁠", "").replace(" ", "")
    book_text = book_text.replace('“', '\"').replace('’', '\'').replace('”', '\"').replace('‘', '\'')

    with open(save_path + "/" + book_title + ".txt", "w") as text_file:
        text_file.write(book_text)