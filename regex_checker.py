import re
text = "boop boop\n1 beep int. bop badoop\n4 il\nhello"
text = text.replace('Boop', 'goop')
print(text)
new_text = re.sub(r"\n[0-9] .*]","", text)

text2 = "\' T"
re.sub(r'(?is)</html>.+', '</html>', article)
print(new_text)

##################################################

# if word starts with I, split, check if second is valid word
#
# from spellchecker import SpellChecker
#
# spell = SpellChecker()
#
# text2 = "Jhave a potato I aye"
# new_text = []
# for word in text2.split(' '):
#     if (word[0] == 'I' or word[0] == 'J' or word[0] == 'T') and len(word) > 3:
#         try:
#             second = word[1:]
#             cor = '' + spell.correction(second)
#             if cor != second:
#                 new_text.append(word)
#             else:
#                 new_text.append('I')
#                 new_text.append(second)
#         except:
#             new_text.append(word)
#     else:
#         new_text.append(word)
#
# new_text = ' '.join(new_text)
# print(new_text)

# # find those words that may be misspelled
# misspelled = spell.unknown(['something', 'is', 'hapenning', 'here'])
#
# for word in misspelled:
#     corr = spell.correction(word)
#     # Get the one `most likely` answer
#     print(spell.correction(word))
#
#     # Get a list of `likely` options
#     print(spell.candidates(word))
