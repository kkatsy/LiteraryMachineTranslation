import pickle


# par3_fp = "/Users/kk/Desktop/LNMT/par3.pkl"
# par3 = pickle.load(open(par3_fp, 'rb'))
#
# lang_to_top = {}
# for key in par3.keys():
#     if len(par3[key]['translator_data']) >= 3:
#         lang = key[-2:]
#         if lang not in lang_to_top:
#             lang_to_top[lang] = {}
#         lang_to_top[lang][key[:-3]] = par3[key]
#
# with open('par3_top.pickle', 'wb') as handle:
#     pickle.dump(lang_to_top, handle, protocol=pickle.HIGHEST_PROTOCOL)

# par3_fp = "par3_top.pickle"
par3 = "/Users/kk/Desktop/LNMT/par3.pkl"
par3 = pickle.load(open(par3, 'rb'))
print()

