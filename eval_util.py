# coding=utf-8
"""
命名实体识别测试工具
"""
from itertools import groupby
try:
    import esm
except:
    print("import esm error.....")

convert = {'S-nh': "A", 'B-nh':'B', 'M-nh':'C', 'E-nh':'D',
           'S-ni': "E", 'B-ni':'F', 'M-ni':'G', 'E-ni':'H',
           'S-ns': "I", 'B-ns': 'J', 'M-ns': 'K', 'E-ns': 'L',
            'O':'M'
           }
names = ["A", "BD", "BCD", "BCCD"]
orgs = ["E", "FH", "FGH", "FGGH", "FGGGH", "FGGGGH", "FGGGGGH", "FGGGGGGH"]
locations = ["I", "JL", "JKL", "JKKL", "JKKKL", "JKKKKL", "JKKKKKL"]
index = esm.Index()
for u in names+orgs+locations:
    index.enter(u)
index.fix()

def find(labels):
    r = index.query(labels)
    r = map(lambda x:[x[0][0], x[0][1]],r)
    valid_index = []
    valid_str = []
    for m, n in groupby(r, key=lambda x:x[0]):
        s = list(n)
        max_length_match = sorted(s, key=lambda x:x[1], reverse=True)[0]
        valid_index.append(max_length_match)
        valid_str.append(labels[max_length_match[0]:max_length_match[1]])
    return valid_index, valid_str

def entity_type(labels, words):
    """
    :param labels:  具体标签
    :param words:  词
    :return:
    """
    labels = map(lambda label: convert[label] if label in convert else convert["O"] ,labels)
    labels = "".join(labels)
    valid_index, valid_str = find(labels)
    entity_t = []
    valid_words = []
    for start, end in valid_index:
        valid_words.append("".join(words[start:end]))
    for id, s in zip(valid_index, valid_str):
       if s in names:
           entity_t.append("name")
       elif s in orgs:
           entity_t.append("org")
       elif s in locations:
           entity_t.append("location")
       else:
           entity_t.append("other")
    return [valid_index, valid_str, entity_t, valid_words]
# print entity_type(['B-nh', 'M-nh', 'E-nh'])
