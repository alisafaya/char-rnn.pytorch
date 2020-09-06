# https://github.com/spro/char-rnn.pytorch

import string
import random
import time
import math
import torch

def read_file(filename):
    file = open(filename).read()
    return file, len(file)

# Turning a string into a tensor
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

# # Turning a string into a tensor
# def char_tensor(string):
#     tensor = torch.zeros(len(string)).long()
#     for c, w in enumerate(string):
#         try:
#             tensor[c] = w2i.get(w, w2i['UNK'])
#         except:
#             continue
#     return tensor

# Readable time elapsed
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Reading and un-unicode-encoding data
# all_characters = "AÂBCDEFGHIÎİJKLMNOPRSTUÛVYZâabcdefghijklmnoprstuûvyzÇÖÜçöüĞğıîŞş1234567890" + string.punctuation + " \t\n\r"
# all_characters = ";0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'\()*+,-./:;<=>?@[]^_`{|}~ \t😂🇬🇵🚿🇧🇯+🇴–[]ー🚨⁦🦠•🤔⁩😷‍👇🤣😳♂🇭🤦=👉า😭่โร💥🏼🔥😀​⚠⁉🙏⃣‼£ค"
all_characters = ";âabcdefghijklmnoprstuûvyzçöüğıîş1234567890" + string.punctuation + " \t\n\r"
n_characters = len(all_characters)

# words, _ = read_file("tr_stems.vocab")
# i2w = words.splitlines()
# w2i = { w : i for i, w in enumerate(i2w) }
# n_characters = len(w2i)