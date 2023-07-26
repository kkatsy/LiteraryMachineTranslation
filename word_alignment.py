
with open("maude.txt", "r") as input:
    input_ = input.read().split("\n")

text = []
for line in input_:
    if line != '':
        text.append(line)

print('hi')
