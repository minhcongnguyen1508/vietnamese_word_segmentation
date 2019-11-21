import pycrfsuite
import sklearn_crfsuite
import ast
import os
from utils import load_n_grams
import pickle
import unicodedata as ud
import sys
import re
import string
import numpy as np

def load_crf_config(filename):
    with open(filename, encoding="utf8") as r:
        params = r.read()
        param_dict = ast.literal_eval(params) # Convert to Dict
    
    return param_dict

def wrapper(func, args):
    """Call a function with argument list"""
    return func(*args)

def load_data_from_file(ab_path):
    chain_words = []
    labels = []
    with open(ab_path, 'r', encoding="utf8") as fr:
        lines = fr.readlines()
        for sentences in lines:
            sentence = []
            sent_labels = []
            for word in sentences.strip().split():
                syllables = word.split("_")
                for i, syllable in enumerate(syllables):
                    sentence.append(syllable)
                    if i == 0:
                        sent_labels.append('B_W')
                    else:
                        sent_labels.append('I_W')
            chain_words.append(sentence)
            labels.append(sent_labels)

    return chain_words, labels

def load_data_from_dir(path):
    file_names = os.listdir(path)
    chain_words = None
    labels = None
    for f_name in file_names:
        file_path = os.path.join(path, f_name)
        if f_name.startswith('.') or os.path.isdir(file_path):
            continue
        batch_sentences, batch_labels = load_data_from_file(file_path)
        if chain_words is None:
            chain_words = batch_sentences
            labels = batch_labels
        else:
            chain_words += batch_sentences
            labels += batch_labels
    return chain_words, labels


class CRF_Segmentation():
    def __init__(self, root_path="", bi_grams_path='bi_grams.txt', tri_grams_path='tri_grams.txt',
                 crf_config_path='crf_config.txt',
                 features_path='crf_features.txt',
                 model_path='vi-word-segment',
                 load_data_f_file=load_data_from_dir,
                 base_lib='sklearn_crfsuite'):
        self.bi_grams = load_n_grams(root_path + bi_grams_path)
        self.tri_grams = load_n_grams(root_path + tri_grams_path)
        self.crf_config = load_crf_config(root_path + crf_config_path)
        self.features_crf_arg = load_crf_config(root_path + features_path)
        self.center_id = int((len(self.features_crf_arg) - 1)/2)
        self.function_dict = {
            'bias': lambda word, *args: 1.0,
            'word.lower()': lambda word, *args: word.lower(),
            'word.isupper()': lambda word, *args: word.isupper(),
            'word.istitle()': lambda word, *args: word.istitle(),
            'word.isdigit()': lambda word, *args: word.isdigit(),
            'word.bi_gram()': lambda word, word1, relative_id, *args: self._check_bi_gram([word, word1], relative_id),
            'word.tri_gram()': lambda word, word1, word2, relative_id, *args: self._check_tri_gram(
                [word, word1, word2], relative_id)
        }
        self.model_path = model_path
        self.load_data_from_file = load_data_f_file
        self.tagger = None
        self.base_lib = base_lib
    
    def _check_bi_gram(self, a, relative_id):
        if relative_id < 0:
            return ' '.join([a[0], a[1]]).lower() in self.bi_grams
        else:
            return ' '.join([a[1], a[0]]).lower() in self.bi_grams

    def _check_tri_gram(self, b, relative_id):
        if relative_id < 0:
            return ' '.join([b[0], b[1], b[2]]).lower() in self.tri_grams
        else:
            return ' '.join([b[2], b[1], b[0]]).lower() in self.tri_grams

    def _get_base_features(self, features_crf_arg, word_list, relative_id=0):
        prefix = ""
        if relative_id < 0:
            prefix = str(relative_id) + ":"
        elif relative_id > 0:
            prefix = '+' + str(relative_id) + ":"

        features_dict = dict()
        for ft_cfg in features_crf_arg:
            features_dict.update({prefix+ft_cfg: wrapper(self.function_dict[ft_cfg], word_list + [relative_id])})
        return features_dict


    def create_syllable_features(self, text, word_id):
        word = text[word_id]
        features_dict = self._get_base_features(self.features_crf_arg[self.center_id], [word])

        if word_id > 0:
            word1 = text[word_id - 1]
            features_dict.update(self._get_base_features(self.features_crf_arg[self.center_id - 1],
                                                         [word1, word], -1))
            if word_id > 1:
                word2 = text[word_id - 2]
                features_dict.update(self._get_base_features(self.features_crf_arg[self.center_id - 2],
                                                             [word2, word1, word], -2))
        if word_id < len(text) - 1:
            word1 = text[word_id + 1]
            features_dict.update(self._get_base_features(self.features_crf_arg[self.center_id + 1],
                                                         [word1, word], +1))
            if word_id < len(text) - 2:
                word2 = text[word_id + 2]
                features_dict.update(self._get_base_features(self.features_crf_arg[self.center_id + 2],
                                                             [word2, word1, word], +2))
        return features_dict

    def create_sentence_features(self, sentences):
        return [self.create_syllable_features(sentences, i) for i in range(len(sentences))]
    
    def prepare_training_data(self, sentences, labels):
        X = []
        y = []
        for i, sentences in enumerate(sentences):
            X.append(self.create_sentence_features(sentences))
            y.append(labels[i])

        return X, y

    def train(self, data_path):
        sentences, labels = self.load_data_from_file(data_path)
        X, y = self.prepare_training_data(sentences, labels)
        if self.base_lib == "sklearn_crfsuite":
            crf = sklearn_crfsuite.CRF(
                algorithm=self.crf_config['algorithm'],
                c1=self.crf_config['c1'],
                c2=self.crf_config['c2'],
                max_iterations=self.crf_config['max_iterations'],
                all_possible_transitions=self.crf_config['all_possible_transitions']
            )
            crf.fit(X, y)
            # joblib.dump(crf, self.model_path)
            with open('../models/'+self.model_path, 'wb') as fw:
                pickle.dump(crf, fw)
        else:
            trainer = pycrfsuite.Trainer(verbose=False)

            for xseq, yseq in zip(X, y):
                trainer.append(xseq, yseq)

            trainer.set_params(self.crf_config)
            trainer.train('../models/'+self.model_path)

    def load_tagger(self):
        print("Loading model from file {}".format('../models/'+self.model_path))
        if self.base_lib == "sklearn_crfsuite":
            # self.tagger = joblib.load(self.model_path)
            with open('../models/'+self.model_path, 'rb') as fr:
                self.tagger = pickle.load(fr)
        else:
            self.tagger = pycrfsuite.Tagger()
            self.tagger.open('../models/'+self.model_path, encoding="utf8")

    @staticmethod
    def syllablize(text):
        text = ud.normalize('NFC', text)
        sign = ["==>", "->", "\.\.\.", ">>"]
        digits = "\d+([\.,_]\d+)+"
        email = "(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
        web = "^(http[s]?://)?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+$"
        datetime = [
            "\d{1,2}\/\d{1,2}(\/\d+)?",
            "\d{1,2}-\d{1,2}(-\d+)?",
        ]
        word = "\w+"
        non_word = "[^\w\s]"
        abbreviations = [
            "[A-ZĐ]+\.",
            "Tp\.",
            "Mr\.", "Mrs\.", "Ms\.",
            "Dr\.", "ThS\."
        ]
        patterns = []
        patterns.extend(abbreviations)
        patterns.extend(sign)
        patterns.extend([web, email])
        patterns.extend(datetime)
        patterns.extend([digits, non_word, word])
        patterns = "(" + "|".join(patterns) + ")"
        if sys.version_info < (3, 0):
            patterns = patterns.decode('utf-8')
        tokens = re.findall(patterns, text, re.UNICODE)
        return [token[0] for token in tokens]

    @staticmethod
    def _check_special_case(word_list):
        # Current word start with upper case but previous word NOT
        if word_list[1].istitle() and (not word_list[0].istitle()):
            return True

        # Is a punctuation
        for word in word_list:
            if word in string.punctuation:
                return True
        # Is a digit
        for word in word_list:
            if word[0].isdigit():
                return True

        return False

    def tokenize(self, text):
        if self.tagger is None:
            self.load_tagger()
        sent = self.syllablize(text)
        syl_len = len(sent)
        if syl_len <= 1:
            return sent
        test_features = self.create_sentence_features(sent)
        if self.base_lib == "sklearn_crfsuite":
            prediction = self.tagger.predict([test_features])[0]
        else:
            prediction = self.tagger.tag(test_features)
        word_list = []
        pre_word = sent[0]
        for i, p in enumerate(prediction[1:], start=1):
            if p == 'I_W' and not self._check_special_case(sent[i-1:i+1]):
                pre_word += "_" + sent[i]
                if i == (syl_len - 1):
                    word_list.append(pre_word)
            else:
                if i > 0:
                    word_list.append(pre_word)
                if i == (syl_len - 1):
                    word_list.append(sent[i])
                pre_word = sent[i]

        return word_list
    
    def get_tokenized(self, text):
        if self.tagger is None:
            self.load_tagger()
        sent = self.syllablize(text)
        if len(sent) <= 1:
            return text
        test_features = self.create_sentence_features(sent)
        if self.base_lib == "sklearn_crfsuite":
            prediction = self.tagger.predict([test_features])[0]
        else:
            prediction = self.tagger.tag(test_features)
        complete = sent[0]
        for i, p in enumerate(prediction[1:], start=1):
            if p == 'I_W' and not self._check_special_case(sent[i-1:i+1]):
                complete = complete + '_' + sent[i]
            else:
                complete = complete + ' ' + sent[i]
        return complete

# Test
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

def test():
    crf_tokenizer_obj = CRF_Segmentation()
    test_sent = u"Tổng thống Nga coi việc Mỹ không kích căn cứ quân sự của Syria là 'sự gây hấn nhằm vào một quốc gia có chủ quyền', gây tổn hại đến quan hệ Moscow-Washington"
    test_sent = test_sent.replace("'", "")
    test_sent = test_sent.replace(",", "")
    test_sent = test_sent.replace("-", " ")
    expect_sent = ["Tổng_thống", "Nga", "coi", "việc", "Mỹ", "không_kích", "căn_cứ", "quân_sự", "của", "Syria", "là", "sự", "gây_hấn", "nhằm", "vào", "một", "quốc_gia", "có", "chủ_quyền", "gây", "tổn_hại", "đến", "quan_hệ", "Moscow", "Washington"]
    # crf_tokenizer_obj.train('../data/tokenized/samples/training')
    tokenized_sent = crf_tokenizer_obj.get_tokenized(test_sent)
    print(tokenized_sent)
    tokens = crf_tokenizer_obj.tokenize(test_sent)
    print(tokens)
    print(expect_sent)
    check_acc(tokens, expect_sent)
    # Test Case 0:
    text = "Kết quả xổ số điện toán Vietlott ngày 6/2/2017"
    actual = crf_tokenizer_obj.tokenize(text)
    expected = ["Kết_quả", "xổ_số", "điện_toán", "Vietlott", "ngày", "6/2/2017"]
    print(actual)
    print(expected)
    check_acc(actual, expected)
    # Test Case 1:
    text1 = "Việt Nam sẽ đảm nhiệm chức Chủ tịch luân phiên của ASEAN vào năm 2020"
    actual1 = crf_tokenizer_obj.tokenize(text1)
    expected1 = ["Việt_Nam", "sẽ", "đảm_nhiệm", "chức", "Chủ_tịch", "luân_phiên", "của", "ASEAN", "vào", "năm", "2020"]
    print(actual1)
    print(expected1)
    check_acc(actual1, expected1)
    # Test Case 2:
    text2 = "Học sinh học sinh học"
    actual2 = crf_tokenizer_obj.tokenize(text2)
    expected2 = ["Học_sinh", "học", "sinh_học"]
    print(actual2)
    print(expected2)
    check_acc(actual2, expected2)

def check_acc(actual, expected):
    expect = np.zeros_like(expected, dtype=bool)
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    for i in range(len(actual)):
        for j in range(len(expected)):
            if(actual[i] == expected[j] and expect[j]==False):
                expect[j] = True
                accuracy += 1
    if accuracy == 0:
        # print("Accuracy: {}%".format(0.01*100), end="\n\n") # True / Sum of word
        f1 = 0
    else:
        recall = accuracy/len(expected)
        precision = accuracy/len(actual)
        print(accuracy, len(expected), len(actual))
        f1 = (2*recall*precision)/(recall+precision)
        # print("Accuracy: {}%".format(f1*100), end="\n\n") # True / Sum of word
    return f1

def test1():
    crf_tokenizer_obj = CRF_Segmentation()
    crf_tokenizer_obj.train('../data/tokenized/samples/training')

    text, expects = load_file_test('../data/tokenized/samples/test/test_data.txt')
    accuracy = []
    result = crf_tokenizer_obj.tokenize(text[3])
    print(result)
    print(expects[3])
    check_acc(result, expects[3])
    for actual, expect in zip(text, expects):
        result = crf_tokenizer_obj.tokenize(actual)
        print(result)
        print(expect)
        accuracy.append(check_acc(result, expect))

    avg = sum(accuracy)/len(accuracy)
    print("F1 Score: {}%".format(avg*100), end="\n\n")

if __name__ == '__main__':
    test1()
