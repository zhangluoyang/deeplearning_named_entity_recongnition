# coding=utf-8
"""
命名实体识别　zhangluoyang 2017-03-13 南京

"""
import numpy as np
import cPickle
import os


class DataSet(object):

    def __init__(self, train_data, test_data, batch_size, context_size, embedding_size, exists=False):
        """

        :param train_data:  训练数据集
        :param test_data: 　测试数据集
        :param batch_size: 　
        :param context_size: 　取的上下文context数量
        :param embedding_size:  词嵌入维数
        :param exists:  数据源是否已经加载完成
        """
        assert context_size%2==1  # 确保context是奇数
        self.single_windows = context_size/2  # 半个context大小
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.batch_size = batch_size
        self.train_data = train_data
        self.test_data = test_data
        if exists:
            print("loading...")
            self.load()
        else:
            print("processing....")
            self.process()
        self.batch_nums = self.num_examples // self.batch_size
        self.index = np.arange(self.num_examples)
        self.point = 0

    def process(self):
        with open(self.train_data, "r") as f:
            train_lines = f.readlines()
            lines = train_lines
            lines.append("")  # 空格符号隔开
        with open(self.test_data, "r") as f:
            test_lines = f.readlines()
            lines = lines + test_lines
        lines = map(lambda line: line.strip(), lines)
        i = 0
        all_words = set()
        all_labels = set()
        instances = []
        for line in lines:
            if i==0:
                i = i + 1
                continue
            elif i==1:
                words_labels = line.split(", ")
                words, labels = [], []
                for word_label in words_labels:
                    word, label = word_label.split("/")
                    if word not in all_words:all_words.add(word)
                    if label not in all_labels:all_labels.add(label)
                    words.append(word)
                    labels.append(label)
                instance = [words, labels]
                instances.append(instance)
                i = i + 1
            elif i==2:
                i = 0
                continue
        all_words.add("unknow")
        all_words = list(all_words)
        all_labels = list(all_labels)
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        words_id = dict(zip(all_words, range(vocab_size)))
        labels_id = dict(zip(all_labels, range(labels_size)))
        if not os.path.exists("data/mlp"):
            os.mkdir("data/mlp")
        cPickle.dump(all_words, open("data/mlp/all_words", "w"))
        cPickle.dump(all_labels, open("data/mlp/all_labels", "w"))
        cPickle.dump(instances, open("data/mlp/instances", "w"))
        cPickle.dump(words_id, open("data/mlp/words_id", "w"))
        cPickle.dump(labels_id, open("data/mlp/labels_id", "w"))
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id
        self.instances = instances
        self.num_examples = len(instances)

    def shuffle(self):
        np.random.shuffle(self.index)
        self.point = 0

    def load(self):
        all_words = cPickle.load(open("data/mlp/all_words", "r"))
        all_labels = cPickle.load(open("data/mlp/all_labels", "r"))
        instances = cPickle.load(open("data/mlp/instances", "r"))
        words_id = cPickle.load(open("data/mlp/words_id", "r"))
        labels_id = cPickle.load(open("data/mlp/labels_id", "r"))
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id
        self.instances = instances
        self.num_examples = len(instances)

    def next_batch(self):
        start = self.point
        self.point = self.point + self.batch_size
        if self.point > self.num_examples:
            self.shuffle()
            start = 0
            self.point = self.point + self.batch_size
        end = self.point
        batch_instances = map(lambda x: self.instances[x], self.index[start:end])  # 获取数据
        batch_context_words = []  # context的词
        batch_target_labels = []  # target的label
        for i in range(len(batch_instances)):
            words, labels = batch_instances[i]
            words = ["unknow"]*self.single_windows + words + ["unknow"]*self.single_windows  # 针对句子的开头和结尾进行padding
            labels = ["O"]*self.single_windows + labels + ["O"]*self.single_windows  #
            for j in range(len(words))[self.single_windows:-self.single_windows]:  # 滑动窗口生成数据
                batch_context_words.append(words[j - self.single_windows:j + self.single_windows + 1])  # 每一次生成一个context_size的长度
                batch_target_labels.append(labels[j])  # target的目标词
        assert len(batch_context_words) == len(batch_target_labels)
        this_batch_size = len(batch_context_words)
        # 数据转换为神经网络的数据格式
        batch_words_id = []
        batch_labels_id = []
        for i in range(this_batch_size):
            words = batch_context_words[i]
            words_id = map(lambda word: self.words_id[word] if word in self.words_id else self.words_id["unknow"], words)
            label = batch_target_labels[i]
            label_id = self.labels_id[label] if label in self.labels_id else self.labels_id["O"]
            one_hot_label_id = [0]*self.labels_size
            one_hot_label_id[label_id] = 1
            batch_words_id.append(words_id)
            batch_labels_id.append(one_hot_label_id)
        assert len(batch_words_id)==len(batch_labels_id)
        return batch_words_id, batch_labels_id


class DataSetEval(object):
    def __init__(self, test_data, batch_size, context_size, embedding_size):
        """
        :param test_data: 　测试数据集
        :param batch_size: 　
        :param context_size: 　取的上下文context数量
        :param embedding_size:  词嵌入维数
        :param exists:  数据源是否已经加载完成
        """
        assert context_size % 2 == 1  # 确保context是奇数
        self.single_windows = context_size / 2  # 半个context大小
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.batch_size = batch_size
        self.test_data = test_data
        self.load()
        self.generate_data()
        self.batch_nums = self.num_examples // self.batch_size
        self.index = np.arange(self.num_examples)
        self.point = 0

    def load(self):
        all_words = cPickle.load(open("data/mlp/all_words", "r"))
        all_labels = cPickle.load(open("data/mlp/all_labels", "r"))
        words_id = cPickle.load(open("data/mlp/words_id", "r"))
        labels_id = cPickle.load(open("data/mlp/labels_id", "r"))
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id

    def generate_data(self):
        with open(self.test_data, "r") as f:
            test_lines = f.readlines()
            lines = test_lines
        lines = map(lambda line: line.strip(), lines)
        i = 0
        instances = []
        for line in lines:
            if i == 0:
                i = i + 1
                continue
            elif i == 1:
                words_labels = line.split(", ")
                words, labels = [], []
                for word_label in words_labels:
                    word, label = word_label.split("/")
                    words.append(word)
                    labels.append(label)
                instance = [words, labels]
                instances.append(instance)
                i = i + 1
            elif i == 2:
                i = 0
                continue
        self.instances = instances
        self.num_examples = len(self.instances)

    def shuffle(self):
        np.random.shuffle(self.index)
        self.point = 0

    def next_batch(self):
        start = self.point
        self.point = self.point + self.batch_size
        if self.point > self.num_examples:
            self.shuffle()
            start = 0
            self.point = self.point + self.batch_size
        end = self.point
        batch_instances = map(lambda x: self.instances[x], self.index[start:end])  # 获取数据
        batch_context_words = []  # context的词
        batch_target_labels = []  # target的label
        for i in range(len(batch_instances)):
            words, labels = batch_instances[i]
            words = ["unknow"]*self.single_windows + words + ["unknow"]*self.single_windows  # 针对句子的开头和结尾进行padding
            labels = ["O"]*self.single_windows + labels + ["O"]*self.single_windows  #
            for j in range(len(words))[self.single_windows:-self.single_windows]:  # 滑动窗口生成数据
                batch_context_words.append(words[j - self.single_windows:j + self.single_windows + 1])  # 每一次生成一个context_size的长度
                batch_target_labels.append(labels[j])  # target的目标词
        assert len(batch_context_words) == len(batch_target_labels)
        this_batch_size = len(batch_context_words)
        # 数据转换为神经网络的数据格式
        batch_words_id = []
        batch_labels_id = []
        for i in range(this_batch_size):
            words = batch_context_words[i]
            words_id = map(lambda word: self.words_id[word] if word in self.words_id else self.words_id["unknow"], words)
            label = batch_target_labels[i]
            label_id = self.labels_id[label] if label in self.labels_id else self.labels_id["O"]
            one_hot_label_id = [0]*self.labels_size
            one_hot_label_id[label_id] = 1
            batch_words_id.append(words_id)
            batch_labels_id.append(one_hot_label_id)
        assert len(batch_words_id)==len(batch_labels_id)
        return batch_words_id, batch_labels_id


class DataSetConvert(object):
    def __init__(self, batch_size, context_size, embedding_size):
        """
        :param test_data: 　测试数据集
        :param batch_size: 　
        :param context_size: 　取的上下文context数量
        :param embedding_size:  词嵌入维数
        :param exists:  数据源是否已经加载完成
        """
        assert context_size % 2 == 1  # 确保context是奇数
        self.single_windows = context_size / 2  # 半个context大小
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.batch_size = batch_size
        self.load()

    def load(self):
        all_words = cPickle.load(open("data/mlp/all_words", "r"))
        all_labels = cPickle.load(open("data/mlp/all_labels", "r"))
        words_id = cPickle.load(open("data/mlp/words_id", "r"))
        labels_id = cPickle.load(open("data/mlp/labels_id", "r"))
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id

    def convert(self, words_labels):
        """

        :param words_labels: 分好词的list
        :return:
        """
        words, labels = [], []
        for word_label in words_labels:
            word, label = word_label.split("/")
            words.append(word)
            labels.append(label)
        words = ["unknow"] * self.single_windows + words + ["unknow"] * self.single_windows
        labels = ["O"] * self.single_windows + labels + ["O"] * self.single_windows  #
        context_words = []
        target_labels = []
        for j in range(len(words))[self.single_windows:-self.single_windows]:  # 滑动窗口生成数据
            context_words.append(words[j - self.single_windows:j + self.single_windows + 1])  # 每一次生成一个context_size的长度
            target_labels.append(labels[j])  # target的目标词
        this_batch_size = len(context_words)
        batch_words_id = []
        batch_labels_id = []
        for i in range(this_batch_size):
            words = context_words[i]
            words_id = map(lambda word: self.words_id[word] if word in self.words_id else self.words_id["unknow"], words)
            label = target_labels[i]
            label_id = self.labels_id[label] if label in self.labels_id else self.labels_id["O"]
            one_hot_label_id = [0] * self.labels_size
            one_hot_label_id[label_id] = 1
            batch_words_id.append(words_id)
            batch_labels_id.append(one_hot_label_id)
        assert len(batch_words_id) == len(batch_labels_id)
        return batch_words_id, batch_labels_id


class DatasetTruned(object):

    def __init__(self, train_data, test_data, batch_size, context_size, embedding_size, word2vec, exists=False):
        """

        :param train_data:  训练数据集
        :param test_data: 　测试数据集
        :param batch_size: 　
        :param context_size: 　取的上下文context数量
        :param embedding_size:  词嵌入维数
        :param word2vec:  预先训练的word2vec dict()
        :param exists:  数据源是否已经加载完成
        """
        assert context_size % 2 == 1  # 确保context是奇数
        self.single_windows = context_size / 2  # 半个context大小
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.batch_size = batch_size
        self.train_data = train_data
        self.test_data = test_data
        self.word2vec =word2vec
        if exists:
            print("loading...")
            self.load()
        else:
            print("processing....")
            self.process()
        self.batch_nums = self.num_examples // self.batch_size
        self.index = np.arange(self.num_examples)
        self.point = 0

    def process(self):
        with open(self.train_data, "r") as f:
            train_lines = f.readlines()
            lines = train_lines
            lines.append("")  # 空格符号隔开
        with open(self.test_data, "r") as f:
            test_lines = f.readlines()
            lines = lines + test_lines
        lines = map(lambda line: line.strip(), lines)
        i = 0
        all_words = set()
        all_labels = set()
        instances = []
        for line in lines:
            if i == 0:
                i = i + 1
                continue
            elif i == 1:
                words_labels = line.split(", ")
                words, labels = [], []
                for word_label in words_labels:
                    word, label = word_label.split("/")
                    if word not in all_words: all_words.add(word)
                    if label not in all_labels: all_labels.add(label)
                    words.append(word)
                    labels.append(label)
                instance = [words, labels]
                instances.append(instance)
                i = i + 1
            elif i == 2:
                i = 0
                continue
        all_words.add("unknow")
        all_words = list(all_words)
        all_labels = list(all_labels)
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        words_id = dict(zip(all_words, range(vocab_size)))
        labels_id = dict(zip(all_labels, range(labels_size)))

        # 生成词向量表
        word_vec = np.zeros(shape=(len(all_words), self.embedding_size), dtype="float32")
        for i in range(len(all_words)):
            word = all_words[i]
            if word in self.word2vec:
                word_vec[i] = self.word2vec[word]
            else:
                word_vec[i] = np.random.rand(1, self.embedding_size)
        self.word_vec = word_vec

        if not os.path.exists("data/mlp_truned"):
            os.mkdir("data/mlp_truned")
        cPickle.dump(all_words, open("data/mlp_truned/all_words", "w"))
        cPickle.dump(all_labels, open("data/mlp_truned/all_labels", "w"))
        cPickle.dump(instances, open("data/mlp_truned/instances", "w"))
        cPickle.dump(words_id, open("data/mlp_truned/words_id", "w"))
        cPickle.dump(labels_id, open("data/mlp_truned/labels_id", "w"))
        cPickle.dump(word_vec, open("data/mlp_truned/word_vec", "w"))
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id
        self.instances = instances
        self.num_examples = len(instances)

    def shuffle(self):
        np.random.shuffle(self.index)
        self.point = 0

    def load(self):
        all_words = cPickle.load(open("data/mlp_truned/all_words", "r"))
        all_labels = cPickle.load(open("data/mlp_truned/all_labels", "r"))
        instances = cPickle.load(open("data/mlp_truned/instances", "r"))
        words_id = cPickle.load(open("data/mlp_truned/words_id", "r"))
        labels_id = cPickle.load(open("data/mlp_truned/labels_id", "r"))
        word_vec = cPickle.load(open("data/mlp_truned/word_vec", "r"))
        self.word_vec = word_vec
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id
        self.instances = instances
        self.num_examples = len(instances)

    def next_batch(self):
        start = self.point
        self.point = self.point + self.batch_size
        if self.point > self.num_examples:
            self.shuffle()
            start = 0
            self.point = self.point + self.batch_size
        end = self.point
        batch_instances = map(lambda x: self.instances[x], self.index[start:end])  # 获取数据
        batch_context_words = []  # context的词
        batch_target_labels = []  # target的label
        for i in range(len(batch_instances)):
            words, labels = batch_instances[i]
            words = ["unknow"] * self.single_windows + words + ["unknow"] * self.single_windows  # 针对句子的开头和结尾进行padding
            labels = ["O"] * self.single_windows + labels + ["O"] * self.single_windows  #
            for j in range(len(words))[self.single_windows:-self.single_windows]:  # 滑动窗口生成数据
                batch_context_words.append(
                    words[j - self.single_windows:j + self.single_windows + 1])  # 每一次生成一个context_size的长度
                batch_target_labels.append(labels[j])  # target的目标词
        assert len(batch_context_words) == len(batch_target_labels)
        this_batch_size = len(batch_context_words)
        # 数据转换为神经网络的数据格式
        batch_words_id = []
        batch_labels_id = []
        for i in range(this_batch_size):
            words = batch_context_words[i]
            words_id = map(lambda word: self.words_id[word] if word in self.words_id else self.words_id["unknow"],
                           words)
            label = batch_target_labels[i]
            label_id = self.labels_id[label] if label in self.labels_id else self.labels_id["O"]
            one_hot_label_id = [0] * self.labels_size
            one_hot_label_id[label_id] = 1
            batch_words_id.append(words_id)
            batch_labels_id.append(one_hot_label_id)
        assert len(batch_words_id) == len(batch_labels_id)
        return batch_words_id, batch_labels_id


class DataSetTrunedEval(object):
    def __init__(self, test_data, batch_size, context_size, embedding_size):
        """
        :param test_data: 　测试数据集
        :param batch_size: 　
        :param context_size: 　取的上下文context数量
        :param embedding_size:  词嵌入维数
        :param exists:  数据源是否已经加载完成
        """
        assert context_size % 2 == 1  # 确保context是奇数
        self.single_windows = context_size / 2  # 半个context大小
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.batch_size = batch_size
        self.test_data = test_data
        self.load()
        self.generate_data()
        self.batch_nums = self.num_examples // self.batch_size
        self.index = np.arange(self.num_examples)
        self.point = 0

    def load(self):
        all_words = cPickle.load(open("data/mlp_truned/all_words", "r"))
        all_labels = cPickle.load(open("data/mlp_truned/all_labels", "r"))
        words_id = cPickle.load(open("data/mlp_truned/words_id", "r"))
        labels_id = cPickle.load(open("data/mlp_truned/labels_id", "r"))
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id

    def generate_data(self):
        with open(self.test_data, "r") as f:
            test_lines = f.readlines()
            lines = test_lines
        lines = map(lambda line: line.strip(), lines)
        i = 0
        instances = []
        for line in lines:
            if i == 0:
                i = i + 1
                continue
            elif i == 1:
                words_labels = line.split(", ")
                words, labels = [], []
                for word_label in words_labels:
                    word, label = word_label.split("/")
                    words.append(word)
                    labels.append(label)
                instance = [words, labels]
                instances.append(instance)
                i = i + 1
            elif i == 2:
                i = 0
                continue
        self.instances = instances
        self.num_examples = len(self.instances)

    def shuffle(self):
        np.random.shuffle(self.index)
        self.point = 0

    def next_batch(self):
        start = self.point
        self.point = self.point + self.batch_size
        if self.point > self.num_examples:
            self.shuffle()
            start = 0
            self.point = self.point + self.batch_size
        end = self.point
        batch_instances = map(lambda x: self.instances[x], self.index[start:end])  # 获取数据
        batch_context_words = []  # context的词
        batch_target_labels = []  # target的label
        for i in range(len(batch_instances)):
            words, labels = batch_instances[i]
            words = ["unknow"] * self.single_windows + words + ["unknow"] * self.single_windows  # 针对句子的开头和结尾进行padding
            labels = ["O"] * self.single_windows + labels + ["O"] * self.single_windows  #
            for j in range(len(words))[self.single_windows:-self.single_windows]:  # 滑动窗口生成数据
                batch_context_words.append(
                    words[j - self.single_windows:j + self.single_windows + 1])  # 每一次生成一个context_size的长度
                batch_target_labels.append(labels[j])  # target的目标词
        assert len(batch_context_words) == len(batch_target_labels)
        this_batch_size = len(batch_context_words)
        # 数据转换为神经网络的数据格式
        batch_words_id = []
        batch_labels_id = []
        for i in range(this_batch_size):
            words = batch_context_words[i]
            words_id = map(lambda word: self.words_id[word] if word in self.words_id else self.words_id["unknow"],
                           words)
            label = batch_target_labels[i]
            label_id = self.labels_id[label] if label in self.labels_id else self.labels_id["O"]
            one_hot_label_id = [0] * self.labels_size
            one_hot_label_id[label_id] = 1
            batch_words_id.append(words_id)
            batch_labels_id.append(one_hot_label_id)
        assert len(batch_words_id) == len(batch_labels_id)
        return batch_words_id, batch_labels_id


class DataSetTrunedConvert(object):
    def __init__(self, batch_size, context_size, embedding_size):
        """
        :param test_data: 　测试数据集
        :param batch_size: 　
        :param context_size: 　取的上下文context数量
        :param embedding_size:  词嵌入维数
        :param exists:  数据源是否已经加载完成
        """
        assert context_size % 2 == 1  # 确保context是奇数
        self.single_windows = context_size / 2  # 半个context大小
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.batch_size = batch_size
        self.load()

    def load(self):
        all_words = cPickle.load(open("data/mlp_truned/all_words", "r"))
        all_labels = cPickle.load(open("data/mlp_truned/all_labels", "r"))
        words_id = cPickle.load(open("data/mlp_truned/words_id", "r"))
        labels_id = cPickle.load(open("data/mlp_truned/labels_id", "r"))
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id

    def convert(self, words_labels):
        """

        :param words_labels: 分好词的list
        :return:
        """
        words, labels = [], []
        for word_label in words_labels:
            word, label = word_label.split("/")
            words.append(word)
            labels.append(label)
        words = ["unknow"] * self.single_windows + words + ["unknow"] * self.single_windows
        labels = ["O"] * self.single_windows + labels + ["O"] * self.single_windows  #
        context_words = []
        target_labels = []
        for j in range(len(words))[self.single_windows:-self.single_windows]:  # 滑动窗口生成数据
            context_words.append(words[j - self.single_windows:j + self.single_windows + 1])  # 每一次生成一个context_size的长度
            target_labels.append(labels[j])  # target的目标词
        this_batch_size = len(context_words)
        batch_words_id = []
        batch_labels_id = []
        for i in range(this_batch_size):
            words = context_words[i]
            words_id = map(lambda word: self.words_id[word] if word in self.words_id else self.words_id["unknow"],
                           words)
            label = target_labels[i]
            label_id = self.labels_id[label] if label in self.labels_id else self.labels_id["O"]
            one_hot_label_id = [0] * self.labels_size
            one_hot_label_id[label_id] = 1
            batch_words_id.append(words_id)
            batch_labels_id.append(one_hot_label_id)
        assert len(batch_words_id) == len(batch_labels_id)
        return batch_words_id, batch_labels_id


def word_to_characters(word):
    """
    :param word: 词
    :return:  返回组成词的字
    """
    characters = []
    for character in word.decode("utf-8"):
        characters.append(character.encode('utf-8'))
    return characters


class DataSeTCharater(object):
    def __init__(self, train_data, test_data, batch_size, context_size, embedding_size, word2vec, character_length, exists=False):
        """

        :param train_data:  训练数据集
        :param test_data: 　测试数据集
        :param batch_size: 　
        :param context_size: 　取的上下文context数量
        :param embedding_size:  词嵌入维数
        :param word2vec:  预先训练的word2vec dict()
        :param character_length: 字的长度
        :param exists:  数据源是否已经加载完成
        """
        assert context_size % 2 == 1  # 确保context是奇数
        self.single_windows = context_size / 2  # 半个context大小
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.batch_size = batch_size
        self.train_data = train_data
        self.test_data = test_data
        self.word2vec = word2vec
        self.character_length = character_length
        if exists:
            print("loading...")
            self.load()
        else:
            print("processing....")
            self.process()
        self.batch_nums = self.num_examples // self.batch_size
        self.index = np.arange(self.num_examples)
        self.point = 0

    def process(self):
        with open(self.train_data, "r") as f:
            train_lines = f.readlines()
            lines = train_lines
            lines.append("")  # 空格符号隔开
        with open(self.test_data, "r") as f:
            test_lines = f.readlines()
            lines = lines + test_lines
        lines = map(lambda line: line.strip(), lines)
        i = 0
        all_words = set()
        all_labels = set()
        instances = []
        for line in lines:
            if i == 0:
                i = i + 1
                continue
            elif i == 1:
                words_labels = line.split(", ")
                words, labels = [], []
                for word_label in words_labels:
                    word, label = word_label.split("/")
                    if word not in all_words: all_words.add(word)
                    if label not in all_labels: all_labels.add(label)
                    words.append(word)
                    labels.append(label)
                instance = [words, labels]
                instances.append(instance)
                i = i + 1
            elif i == 2:
                i = 0
                continue
        all_words.add("unknow")
        all_words = list(all_words)
        all_labels = list(all_labels)
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        words_id = dict(zip(all_words, range(vocab_size)))
        labels_id = dict(zip(all_labels, range(labels_size)))
        # 生成字
        all_character = set()
        for word in all_words:
            for c in word_to_characters(word):
                all_character.add(c)
        all_character.add('*')
        all_character = list(all_character)
        character_size = len(all_character)
        characters_id = dict(zip(all_character, range(character_size)))
        # 生成词向量表
        word_vec = np.zeros(shape=(len(all_words), self.embedding_size), dtype="float32")
        for i in range(len(all_words)):
            word = all_words[i]
            if word in self.word2vec:
                word_vec[i] = self.word2vec[word]
            else:
                word_vec[i] = np.random.rand(1, self.embedding_size)
        self.word_vec = word_vec

        if not os.path.exists("data/mlp_truned_character"):
            os.mkdir("data/mlp_truned_character")
        cPickle.dump(all_words, open("data/mlp_truned_character/all_words", "w"))
        cPickle.dump(all_labels, open("data/mlp_truned_character/all_labels", "w"))
        cPickle.dump(instances, open("data/mlp_truned_character/instances", "w"))
        cPickle.dump(words_id, open("data/mlp_truned_character/words_id", "w"))
        cPickle.dump(labels_id, open("data/mlp_truned_character/labels_id", "w"))
        cPickle.dump(word_vec, open("data/mlp_truned_character/word_vec", "w"))
        cPickle.dump(all_character, open("data/mlp_truned_character/all_character", "w"))
        cPickle.dump(characters_id, open("data/mlp_truned_character/characters_id", "w"))


        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id
        self.all_character = all_character
        self.characters_id = characters_id
        self.character_size = character_size
        self.instances = instances
        self.num_examples = len(instances)

    def shuffle(self):
        np.random.shuffle(self.index)
        self.point = 0

    def load(self):
        all_words = cPickle.load(open("data/mlp_truned_character/all_words", "r"))
        all_labels = cPickle.load(open("data/mlp_truned_character/all_labels", "r"))
        instances = cPickle.load(open("data/mlp_truned_character/instances", "r"))
        words_id = cPickle.load(open("data/mlp_truned_character/words_id", "r"))
        labels_id = cPickle.load(open("data/mlp_truned_character/labels_id", "r"))
        word_vec = cPickle.load(open("data/mlp_truned_character/word_vec", "r"))
        characters_id = cPickle.load(open("data/mlp_truned_character/characters_id", "r"))
        all_character = cPickle.load(open("data/mlp_truned_character/all_character", "r"))
        self.characters_id = characters_id
        self.all_character = all_character
        self.character_size = len(all_character)
        self.word_vec = word_vec
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id
        self.instances = instances
        self.num_examples = len(instances)

    def next_batch(self):
        start = self.point
        self.point = self.point + self.batch_size
        if self.point > self.num_examples:
            self.shuffle()
            start = 0
            self.point = self.point + self.batch_size
        end = self.point
        batch_instances = map(lambda x: self.instances[x], self.index[start:end])  # 获取数据
        batch_context_words = []  # context的词
        batch_target_labels = []  # target的label
        for i in range(len(batch_instances)):
            words, labels = batch_instances[i]
            words = ["unknow"] * self.single_windows + words + ["unknow"] * self.single_windows  # 针对句子的开头和结尾进行padding
            labels = ["O"] * self.single_windows + labels + ["O"] * self.single_windows  #
            for j in range(len(words))[self.single_windows:-self.single_windows]:  # 滑动窗口生成数据
                batch_context_words.append(
                    words[j - self.single_windows:j + self.single_windows + 1])  # 每一次生成一个context_size的长度
                batch_target_labels.append(labels[j])  # target的目标词
        assert len(batch_context_words) == len(batch_target_labels)
        this_batch_size = len(batch_context_words)
        # 数据转换为神经网络的数据格式
        batch_words_id = []
        batch_labels_id = []
        batch_characters_id = []
        for i in range(this_batch_size):
            words = batch_context_words[i]
            ws_characters = map(lambda word: word_to_characters(word), words)
            w_cs = []  # 获取字级别的特征
            for w_c in ws_characters:
                if len(w_c) > self.character_length:
                    s = w_c[-self.character_length:]     # 长度过长
                else:
                    s = ['*']*(self.character_length - len(w_c))+w_c  # 长度不足
                s_id = map(lambda w_c: self.characters_id[w_c] if w_c in self.characters_id else self.characters_id['*'], s)  # 将字转换为字的id
                w_cs.append(s_id)
            batch_characters_id.append(w_cs)
            words_id = map(lambda word: self.words_id[word] if word in self.words_id else self.words_id["unknow"], words)
            label = batch_target_labels[i]
            label_id = self.labels_id[label] if label in self.labels_id else self.labels_id["O"]
            one_hot_label_id = [0] * self.labels_size
            one_hot_label_id[label_id] = 1
            batch_words_id.append(words_id)
            batch_labels_id.append(one_hot_label_id)
        assert len(batch_words_id) == len(batch_labels_id)==len(batch_characters_id)
        return batch_words_id, batch_labels_id, batch_characters_id


class DataSeTCharaterEval(object):

    def __init__(self, test_data, context_size, embedding_size, character_length, batch_size):
        """

        :param train_data:  训练数据集
        :param test_data: 　测试数据集
        :param batch_size: 　
        :param context_size: 　取的上下文context数量
        :param embedding_size:  词嵌入维数
        :param word2vec:  预先训练的word2vec dict()
        :param character_length: 字的长度
        :param exists:  数据源是否已经加载完成
        """
        assert context_size % 2 == 1  # 确保context是奇数
        self.single_windows = context_size / 2  # 半个context大小
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.test_data = test_data
        self.character_length = character_length
        self.batch_size = batch_size
        self.load()
        self.generate_data()
        self.batch_nums = self.num_examples // self.batch_size
        self.index = np.arange(self.num_examples)
        self.point = 0

    def generate_data(self):
        with open(self.test_data, "r") as f:
            test_lines = f.readlines()
            lines = test_lines
        lines = map(lambda line: line.strip(), lines)
        i = 0
        instances = []
        for line in lines:
            if i == 0:
                i = i + 1
                continue
            elif i == 1:
                words_labels = line.split(", ")
                words, labels = [], []
                for word_label in words_labels:
                    word, label = word_label.split("/")
                    words.append(word)
                    labels.append(label)
                instance = [words, labels]
                instances.append(instance)
                i = i + 1
            elif i == 2:
                i = 0
                continue
        self.instances = instances
        self.num_examples = len(self.instances)

    def shuffle(self):
        np.random.shuffle(self.index)
        self.point = 0

    def load(self):
        all_words = cPickle.load(open("data/mlp_truned_character/all_words", "r"))
        all_labels = cPickle.load(open("data/mlp_truned_character/all_labels", "r"))
        instances = cPickle.load(open("data/mlp_truned_character/instances", "r"))
        words_id = cPickle.load(open("data/mlp_truned_character/words_id", "r"))
        labels_id = cPickle.load(open("data/mlp_truned_character/labels_id", "r"))
        word_vec = cPickle.load(open("data/mlp_truned_character/word_vec", "r"))
        characters_id = cPickle.load(open("data/mlp_truned_character/characters_id", "r"))
        all_character = cPickle.load(open("data/mlp_truned_character/all_character", "r"))
        self.characters_id = characters_id
        self.all_character = all_character
        self.character_size = len(all_character)
        self.word_vec = word_vec
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id
        self.instances = instances
        self.num_examples = len(instances)

    def next_batch(self):
        start = self.point
        self.point = self.point + self.batch_size
        if self.point > self.num_examples:
            self.shuffle()
            start = 0
            self.point = self.point + self.batch_size
        end = self.point
        batch_instances = map(lambda x: self.instances[x], self.index[start:end])  # 获取数据
        batch_context_words = []  # context的词
        batch_target_labels = []  # target的label
        for i in range(len(batch_instances)):
            words, labels = batch_instances[i]
            words = ["unknow"] * self.single_windows + words + ["unknow"] * self.single_windows  # 针对句子的开头和结尾进行padding
            labels = ["O"] * self.single_windows + labels + ["O"] * self.single_windows  #
            for j in range(len(words))[self.single_windows:-self.single_windows]:  # 滑动窗口生成数据
                batch_context_words.append(
                    words[j - self.single_windows:j + self.single_windows + 1])  # 每一次生成一个context_size的长度
                batch_target_labels.append(labels[j])  # target的目标词
        assert len(batch_context_words) == len(batch_target_labels)
        this_batch_size = len(batch_context_words)
        # 数据转换为神经网络的数据格式
        batch_words_id = []
        batch_labels_id = []
        batch_characters_id = []
        for i in range(this_batch_size):
            words = batch_context_words[i]
            ws_characters = map(lambda word: word_to_characters(word), words)
            w_cs = []  # 获取字级别的特征
            for w_c in ws_characters:
                if len(w_c) > self.character_length:
                    s = w_c[-self.character_length:]     # 长度过长
                else:
                    s = ['*']*(self.character_length - len(w_c))+w_c  # 长度不足
                s_id = map(lambda w_c: self.characters_id[w_c] if w_c in self.characters_id else self.characters_id['*'], s)  # 将字转换为字的id
                w_cs.append(s_id)
            batch_characters_id.append(w_cs)
            words_id = map(lambda word: self.words_id[word] if word in self.words_id else self.words_id["unknow"], words)
            label = batch_target_labels[i]
            label_id = self.labels_id[label] if label in self.labels_id else self.labels_id["O"]
            one_hot_label_id = [0] * self.labels_size
            one_hot_label_id[label_id] = 1
            batch_words_id.append(words_id)
            batch_labels_id.append(one_hot_label_id)
        assert len(batch_words_id) == len(batch_labels_id)==len(batch_characters_id)
        return batch_words_id, batch_labels_id, batch_characters_id


class DataSeTCharaterConvert(object):

    def __init__(self, context_size, embedding_size, character_length):
        """

        :param train_data:  训练数据集
        :param test_data: 　测试数据集
        :param batch_size: 　
        :param context_size: 　取的上下文context数量
        :param embedding_size:  词嵌入维数
        :param word2vec:  预先训练的word2vec dict()
        :param character_length: 字的长度
        :param exists:  数据源是否已经加载完成
        """
        assert context_size % 2 == 1  # 确保context是奇数
        self.single_windows = context_size / 2  # 半个context大小
        self.embedding_size = embedding_size
        self.context_size = context_size
        self.character_length = character_length
        self.load()

    def load(self):
        all_words = cPickle.load(open("data/mlp_truned_character/all_words", "r"))
        all_labels = cPickle.load(open("data/mlp_truned_character/all_labels", "r"))
        instances = cPickle.load(open("data/mlp_truned_character/instances", "r"))
        words_id = cPickle.load(open("data/mlp_truned_character/words_id", "r"))
        labels_id = cPickle.load(open("data/mlp_truned_character/labels_id", "r"))
        word_vec = cPickle.load(open("data/mlp_truned_character/word_vec", "r"))
        characters_id = cPickle.load(open("data/mlp_truned_character/characters_id", "r"))
        all_character = cPickle.load(open("data/mlp_truned_character/all_character", "r"))
        self.characters_id = characters_id
        self.all_character = all_character
        self.character_size = len(all_character)
        self.word_vec = word_vec
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id
        self.instances = instances
        self.num_examples = len(instances)

    def convert(self, words_labels):
        """

        :param words_labels: 分好词的list
        :return:
        """
        words, labels = [], []
        for word_label in words_labels:
            word, label = word_label.split("/")
            words.append(word)
            labels.append(label)
        words = ["unknow"] * self.single_windows + words + ["unknow"] * self.single_windows
        labels = ["O"] * self.single_windows + labels + ["O"] * self.single_windows  #
        context_words = []
        target_labels = []
        for j in range(len(words))[self.single_windows:-self.single_windows]:  # 滑动窗口生成数据
            context_words.append(words[j - self.single_windows:j + self.single_windows + 1])  # 每一次生成一个context_size的长度
            target_labels.append(labels[j])  # target的目标词
        this_batch_size = len(context_words)
        batch_words_id = []
        batch_labels_id = []
        batch_characters_id = []
        for i in range(this_batch_size):
            words = context_words[i]
            ws_characters = map(lambda word: word_to_characters(word), words)
            w_cs = []  # 获取字级别的特征
            for w_c in ws_characters:
                if len(w_c) > self.character_length:
                    s = w_c[-self.character_length:]  # 长度过长
                else:
                    s = ['*'] * (self.character_length - len(w_c)) + w_c  # 长度不足
                s_id = map(
                    lambda w_c: self.characters_id[w_c] if w_c in self.characters_id else self.characters_id['*'],
                    s)  # 将字转换为字的id
                w_cs.append(s_id)
            batch_characters_id.append(w_cs)
            words_id = map(lambda word: self.words_id[word] if word in self.words_id else self.words_id["unknow"], words)
            label = target_labels[i]
            label_id = self.labels_id[label] if label in self.labels_id else self.labels_id["O"]
            one_hot_label_id = [0] * self.labels_size
            one_hot_label_id[label_id] = 1
            batch_words_id.append(words_id)
            batch_labels_id.append(one_hot_label_id)
        assert len(batch_words_id) == len(batch_labels_id)==len(batch_characters_id)
        return batch_words_id, batch_labels_id, batch_characters_id