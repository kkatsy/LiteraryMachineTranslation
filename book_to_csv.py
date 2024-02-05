import csv
import pandas as pd
import glob

# path = 'russian_lit/russian_language_text'
# save_path = 'russian_lit/russian_language_csv'

# files = glob.glob(path + '/*.txt')
# for f in files:
#     book_title = (f.split('/')[-1]).split('.')[0]
#     with open(f, "r") as fp:
#         book_text = fp.read()
#
#     book_pars = book_text.split('\n')
#     book_pars = [par for par in book_pars if par != '']
#
#     df = pd.DataFrame(book_pars)
#     df.to_csv(save_path + '/' + book_title + '.csv', index = False)


path = 'russian_lit/ru_en_csv'
save_path_en = 'russian_lit/en_gt'
save_path_ru = 'russian_lit/ru_gt'

files = glob.glob(path + '/*.csv')
for f in files:
    book_title = (f.split('/')[-1]).split('_')[0]
    with open(f, "r") as fp:
        book_text = fp.read()

    df = pd.read_csv(f)
    ru_pars = df['0'].tolist()
    en_pars = df['1'].tolist()

    with open(save_path_ru + '/' + book_title + '_ru.txt', "w") as r:
        for par in ru_pars:
            r.write(f"\n{par}")

    with open(save_path_en + '/' + book_title + '_en.txt', "w") as r:
        for par in en_pars:
            r.write(f"\n{par}")



