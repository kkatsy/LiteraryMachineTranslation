with open("russian_lit/ru_books_text_cleaner/non_copyrighted/TheBrothersKaramazov_Garnett.txt", "r") as fp:
    book_text = fp.read()

book_text = book_text.replace("⁠", "").replace(" ", "")

with open("russian_lit/ru_books_text_cleaner/non_copyrighted/TheBrothersKaramazov_Garnett.txt", "w") as text_file:
    text_file.write(book_text)