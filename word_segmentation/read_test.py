
def load_file_test(ab_path):
    sentences = []
    expects = []
    with open(ab_path, 'r', encoding="utf8") as fr:
        lines = fr.readlines()
        for i in lines:
            expect = clean_text(i)
            expects.append(expect)
            sent = prepare_input(i)
            sentences.append(sent)
            
    return sentences, expects
def clean_text(sentence):
    i = sentence
    i = i.replace("-", "")
    i = i.replace("*", "")
    i = i.replace(",", "")
    i = i.replace(".", "")
    i = i.replace("(", "")
    i = i.replace(")", "")
    i = i.replace("\n", "")
    sentence = i.split(' ')
    for i in sentence:
        if i =='':
            sentence.remove('')
    return sentence
def prepare_input(sentence):
    i = sentence
    i = i.replace("-", "")
    i = i.replace("*", "")
    i = i.replace(",", "")
    i = i.replace(".", "")
    i = i.replace("(", "")
    i = i.replace(")", "")
    i = i.replace("\n", "")
    sentence = i.replace('_', " ")
    # print(sentence)
    return sentence
# sentences, expects = load_file_test('../data/tokenized/samples/test/data.txt')
# print(sentences[0])
# print(expects[0])
a = [1, 2, 3, 4, 5]
b = [1, 3, 5, 7, 9]
for x, y in zip(a, b):
    print(x, y)