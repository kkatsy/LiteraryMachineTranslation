from PIL import Image
import glob
import pytesseract
import os
from tqdm import tqdm
save_dir = 'russian_lit/ru_books_text_raw'

data_dir = "/Users/kk/Desktop/book_scans"
book_subdirs = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]


for book in book_subdirs:
    book_path = data_dir + '/' + book
    book_text_file = save_dir + '/' + book + '.txt'
    if not os.path.isfile(book_text_file):
        print('OCRing ' + book, ':')
        text_list = []

        files = glob.glob(book_path + '/*.jpg')
        files_sorted = sorted(files, key=lambda x: int(x.partition('Page ')[2].partition('.jpg')[0]))
        # count = 0
        for filename in tqdm(files_sorted):
            im = Image.open(filename)

            extracted_text = pytesseract.image_to_string(im)
            text_list.append(extracted_text)
            # text_list.append('PAGEEND')
            print('\n Scanned: ', filename)
            # count += 1
            # if count == 3:
            #     break

        text = '\n'.join(text_list)
        with open(book_text_file, "w") as text_file:
            text_file.write(text)


