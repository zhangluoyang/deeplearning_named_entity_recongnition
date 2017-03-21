# coding=utf-8
"""
命名实体识别　zhangluoyang 2017-03-13 南京
"""
from eval_util import *
import numpy as np
import tensorflow as tf
import logging
import os


class mlp_model(object):

    def __init__(self, context_size, word_embedding_size, vocab_size, label_size, batch_size, hidden_layer, lr=0.0003, l2_reg_lambda=0.001, model_name="mlp_model"):
        """

        :param context_size:   窗口大小
        :param word_embedding_size:   训练集当中词的embedding_size
        :param vocab_size:  训练集中词数目
        :param label_size:   标注数目
        :param batch_size:
        :param hidden_layer:   隐藏层神经元数目
        :param lr:  学习率
        :param l2_reg_lambda:　　正则化参数
        :param model_name: 　模型，名称
        """
        self.model_name = model_name
        self.word_embedding_size = word_embedding_size
        self.lr = lr
        self.hidden_layer = hidden_layer
        self.label_size = label_size
        # [batch_size, context_size]
        self.input_word_id = tf.placeholder(dtype=tf.int32, shape=(batch_size, context_size), name="input_words_id")
        # [batch_size, context_size]
        self.input_labels = tf.placeholder(dtype=tf.int32, shape=(batch_size, label_size), name="input_labels")
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)  # 正则化损失

        with tf.name_scope("word_embedding"):
            # [vocab_size, word_embedding_size]
            W = tf.Variable(tf.random_normal(shape=[vocab_size, word_embedding_size], stddev=0.01), name="W")
            # [batch_size, context_size, word_embedding]
            word_embedding_features = tf.nn.embedding_lookup(W, self.input_word_id)
            l2_loss += tf.nn.l2_loss(word_embedding_features)
        features = tf.reshape(word_embedding_features, shape=(-1, context_size * (word_embedding_size)))
        with tf.name_scope("hidden_layer"):
            W = tf.Variable(tf.random_normal(shape=[context_size * (word_embedding_size ), hidden_layer], stddev=0.01), name="W")
            b = tf.Variable(tf.random_normal(shape=[hidden_layer], stddev=0.01), name="b")
            xWb = tf.nn.xw_plus_b(features, W, b)
            hidden_layer = tf.nn.relu(xWb)
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
        # dropout层
        with tf.name_scope("dropout"):
            dropout = tf.nn.dropout(hidden_layer, self.dropout_keep_prob)
        # 输出层
        with tf.name_scope("output"):
            W = tf.Variable(tf.random_normal(shape=[self.hidden_layer, label_size],stddev=0.01), name="W")
            b = tf.Variable(tf.random_normal(shape=[label_size], stddev=0.01), name="b")
            xWb = tf.nn.xw_plus_b(dropout, W, b)
            self.scores = tf.nn.softmax(xWb, name="scores")
            l2_loss += tf.nn.l2_loss(W)  # 计算L2正则化参数
            l2_loss += tf.nn.l2_loss(b)
            self.predictions = tf.argmax(xWb, 1, name="predictions")
            loss = tf.nn.softmax_cross_entropy_with_logits(xWb, self.input_labels)
            self.loss = tf.reduce_mean(loss) + l2_reg_lambda * l2_loss
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_labels, 1))
            self.accuracys = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracys")

    def fit(self, datas, epochs, keep_prob=0.5):
        import datetime
        logging.basicConfig(filename="{}.log".format(self.model_name), format="%(message)s", filemode="a",
                            level=logging.INFO)
        checkpoint_dir = os.path.abspath(os.path.join(self.model_name, "checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        with tf.Session() as sess:
            self.sess = sess
            optimizer = tf.train.AdamOptimizer(self.lr)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            train_op = optimizer.apply_gradients(grads_and_vars)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)
            sess.run(tf.initialize_all_variables())
            def train_step(x_batch, y_batch):
                feed_dict = {self.input_word_id: x_batch, self.input_labels: y_batch, self.dropout_keep_prob: keep_prob}
                _, loss, accuracys = sess.run([train_op, self.loss, self.accuracys], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                logging.info("{}: loss: {}, acc: {}".format(time_str, loss, accuracys))

            for i in range(epochs):
                datas.shuffle()
                for _ in range(datas.batch_nums):
                    x_batch, y_batch = datas.next_batch()
                    train_step(x_batch, y_batch)
                saver.save(sess, checkpoint_prefix, global_step=i)  # 存储训练数据

    def eval(self, datas):
        def eval_step(x_batch, y_batch):
            feed_dict = {self.input_word_id: x_batch, self.input_labels: y_batch, self.dropout_keep_prob: 1.0}
            accuracys, predictions = self.sess.run([self.accuracys, self.predictions], feed_dict)
            return accuracys, predictions
        eval_acc = []
        for _ in range(datas.batch_nums):
            x_batch, y_batch = datas.next_batch()
            acc, pres = eval_step(x_batch=x_batch, y_batch=y_batch)
            eval_acc.append(acc)
        print np.mean(eval_acc)

    def load(self, checkpointfile):
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
            self.sess = sess
            saver = tf.train.import_meta_graph("{}.meta".format(checkpointfile))
            saver.restore(sess, checkpointfile)
            self.input_word_id = graph.get_operation_by_name("input_words_id").outputs[0]
            self.input_labels = graph.get_operation_by_name("input_labels").outputs[0]
            self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            self.accuracys = graph.get_operation_by_name("accuracy/accuracys").outputs[0]
            self.scores = graph.get_operation_by_name("output/scores").outputs[0]

    def predict(self, input_x):
        feed_dict = {self.input_word_id: input_x, self.dropout_keep_prob: 1.0}
        predictions, scores = self.sess.run([self.predictions, self.scores], feed_dict=feed_dict)
        return predictions, scores


class mlp_model_truned(object):

    def __init__(self, context_size, word_embedding_size, vocab_size, label_size, batch_size, hidden_layer, word2vec_init, lr=0.0003, l2_reg_lambda=0.001, model_name="mlp_model_truned"):
        """

        :param context_size:   窗口大小
        :param word_embedding_size:   训练集当中词的embedding_size
        :param vocab_size:  训练集中词数目
        :param label_size:   标注数目
        :param batch_size:
        :param hidden_layer:   隐藏层神经元数目
        :param word2vec_init:  word2vec预先训练的向量
        :param lr:  学习率
        :param l2_reg_lambda:　　正则化参数
        :param model_name: 　模型，名称
        """
        self.model_name = model_name
        self.word_embedding_size = word_embedding_size
        self.lr = lr
        self.hidden_layer = hidden_layer
        self.label_size = label_size
        # [batch_size, context_size]
        self.input_word_id = tf.placeholder(dtype=tf.int32, shape=(batch_size, context_size), name="input_words_id")
        # [batch_size, context_size]
        self.input_labels = tf.placeholder(dtype=tf.int32, shape=(batch_size, label_size), name="input_labels")
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)  # 正则化损失
        with tf.name_scope("word_embedding"):
            # [vocab_size, word_embedding_size]
            W = tf.Variable(word2vec_init, name="W")
            # [batch_size, context_size, word_embedding]
            word_embedding_features = tf.nn.embedding_lookup(W, self.input_word_id)
            l2_loss += tf.nn.l2_loss(word_embedding_features)
        features = tf.reshape(word_embedding_features, shape=(-1, context_size * (word_embedding_size)))
        with tf.name_scope("hidden_layer"):
            W = tf.Variable(tf.random_normal(shape=[context_size * (word_embedding_size ), hidden_layer], stddev=0.01), name="W")
            b = tf.Variable(tf.random_normal(shape=[hidden_layer], stddev=0.01), name="b")
            xWb = tf.nn.xw_plus_b(features, W, b)
            hidden_layer = tf.nn.relu(xWb)
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
        # dropout层
        with tf.name_scope("dropout"):
            dropout = tf.nn.dropout(hidden_layer, self.dropout_keep_prob)
        # 输出层
        with tf.name_scope("output"):
            W = tf.Variable(tf.random_normal(shape=[self.hidden_layer, label_size],stddev=0.01), name="W")
            b = tf.Variable(tf.random_normal(shape=[label_size], stddev=0.01), name="b")
            xWb = tf.nn.xw_plus_b(dropout, W, b)
            self.scores = tf.nn.softmax(xWb, name="scores")
            l2_loss += tf.nn.l2_loss(W)  # 计算L2正则化参数
            l2_loss += tf.nn.l2_loss(b)
            self.predictions = tf.argmax(xWb, 1, name="predictions")
            loss = tf.nn.softmax_cross_entropy_with_logits(xWb, self.input_labels)
            self.loss = tf.reduce_mean(loss) + l2_reg_lambda * l2_loss
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_labels, 1))
            self.accuracys = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracys")

    def fit(self, datas, epochs, keep_prob=0.5):
        import datetime
        logging.basicConfig(filename="{}.log".format(self.model_name), format="%(message)s", filemode="a",
                            level=logging.INFO)
        checkpoint_dir = os.path.abspath(os.path.join(self.model_name, "checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        with tf.Session() as sess:
            self.sess = sess
            optimizer = tf.train.AdamOptimizer(self.lr)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            train_op = optimizer.apply_gradients(grads_and_vars)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)
            sess.run(tf.initialize_all_variables())

            def train_step(x_batch, y_batch):
                feed_dict = {self.input_word_id: x_batch, self.input_labels: y_batch, self.dropout_keep_prob: keep_prob}
                _, loss, accuracys = sess.run([train_op, self.loss, self.accuracys], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                logging.info("{}: loss: {}, acc: {}".format(time_str, loss, accuracys))

            for i in range(epochs):
                datas.shuffle()
                for _ in range(datas.batch_nums):
                    x_batch, y_batch = datas.next_batch()
                    train_step(x_batch, y_batch)
                saver.save(sess, checkpoint_prefix, global_step=i)  # 存储训练数据

    def eval(self, datas):
        def eval_step(x_batch, y_batch):
            feed_dict = {self.input_word_id: x_batch, self.input_labels: y_batch, self.dropout_keep_prob: 1.0}
            accuracys, predictions = self.sess.run([self.accuracys, self.predictions], feed_dict)
            return accuracys, predictions

        eval_acc = []
        for _ in range(datas.batch_nums):
            x_batch, y_batch = datas.next_batch()
            acc, pres = eval_step(x_batch=x_batch, y_batch=y_batch)
            eval_acc.append(acc)
        print np.mean(eval_acc)

    def load(self, checkpointfile):
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
            self.sess = sess
            saver = tf.train.import_meta_graph("{}.meta".format(checkpointfile))
            saver.restore(sess, checkpointfile)
            self.input_word_id = graph.get_operation_by_name("input_words_id").outputs[0]
            self.input_labels = graph.get_operation_by_name("input_labels").outputs[0]
            self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            self.accuracys = graph.get_operation_by_name("accuracy/accuracys").outputs[0]
            self.scores = graph.get_operation_by_name("output/scores").outputs[0]

    def predict(self, input_x):
        feed_dict = {self.input_word_id: input_x, self.dropout_keep_prob: 1.0}
        predictions, scores = self.sess.run([self.predictions, self.scores], feed_dict=feed_dict)
        return predictions, scores


class mlp_truned_character_model(object):

    def __init__(self, context_size, word_embedding_size, vocab_size, label_size, batch_size, hidden_layer, word2vec_init, character_length, character_size, character_embedding,
                 filter_nums, filter_length, lr=0.0003, l2_reg_lambda=0.001, model_name="mlp_truned_character_model"):
        """
        这里使用字级别的卷积　可以用于
        :param context_size:   窗口大小
        :param word_embedding_size:   训练集当中词的embedding_size
        :param vocab_size:  训练集中词数目
        :param label_size:   标注数目
        :param batch_size:
        :param hidden_layer:   隐藏层神经元数目
        :param word2vec_init:  word2vec预先训练的向量
        :param character_length:  字特征的长度
        :param character_size:  训练集当中字的数目
        :param character_embedding:  字向量的嵌入维数
        :param filter_nums: 滤波器数目
        :param filter_length: 卷积核的长度
        :param lr:  学习率
        :param l2_reg_lambda:　　正则化参数
        :param model_name: 　模型，名称
        """
        self.model_name = model_name
        self.word_embedding_size = word_embedding_size
        self.lr = lr
        self.hidden_layer = hidden_layer
        self.label_size = label_size
        # [batch_size, context_size]
        self.input_word_id = tf.placeholder(dtype=tf.int32, shape=(batch_size, context_size), name="input_words_id")
        # [batch_size, context_size, character_length]
        self.input_characters_id = tf.placeholder(dtype=tf.int32, shape=(batch_size, context_size, character_length), name="input_characters_id")
        # [batch_size, context_size]
        self.input_labels = tf.placeholder(dtype=tf.int32, shape=(batch_size, label_size), name="input_labels")
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)  # 正则化损失
        with tf.name_scope("character_embedding"):
            W = tf.Variable(tf.random_normal(shape=[character_size, character_embedding], stddev=0.01), name="W")
            character_embedding_features = tf.nn.embedding_lookup(W, self.input_characters_id)
            character_embedding_features_ = tf.reshape(character_embedding_features, shape=[-1, character_length, character_embedding, 1])
        with tf.name_scope("convolution_layer"):
            filter_shape = [filter_length, character_embedding, 1, filter_nums]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[filter_nums]), name="b")
            conv = tf.nn.conv2d(character_embedding_features_, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # 字特征
            pooled = tf.nn.max_pool(h, ksize=[1, character_length-filter_length+1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
            # [batch_size, context_size*filter_nums]
            character_features = tf.reshape(pooled, shape=[-1, context_size*filter_nums])
        with tf.name_scope("word_embedding"):
            # [vocab_size, word_embedding_size]
            W = tf.Variable(word2vec_init, name="W")
            # [batch_size, context_size, word_embedding]
            word_embedding_features = tf.nn.embedding_lookup(W, self.input_word_id)
            l2_loss += tf.nn.l2_loss(word_embedding_features)
        features = tf.reshape(word_embedding_features, shape=(-1, context_size * (word_embedding_size)))
        input_features = tf.concat(1, values=(features, character_features))
        with tf.name_scope("hidden_layer"):
            W = tf.Variable(tf.random_normal(shape=[context_size * (word_embedding_size + filter_nums), hidden_layer], stddev=0.01), name="W")
            b = tf.Variable(tf.random_normal(shape=[hidden_layer], stddev=0.01), name="b")
            xWb = tf.nn.xw_plus_b(input_features, W, b)
            hidden_layer = tf.nn.relu(xWb)
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
        # dropout层
        with tf.name_scope("dropout"):
            dropout = tf.nn.dropout(hidden_layer, self.dropout_keep_prob)
        # 输出层
        with tf.name_scope("output"):
            W = tf.Variable(tf.random_normal(shape=[self.hidden_layer, label_size],stddev=0.01), name="W")
            b = tf.Variable(tf.random_normal(shape=[label_size], stddev=0.01), name="b")
            xWb = tf.nn.xw_plus_b(dropout, W, b)
            self.scores = tf.nn.softmax(xWb, name="scores")
            l2_loss += tf.nn.l2_loss(W)  # 计算L2正则化参数
            l2_loss += tf.nn.l2_loss(b)
            self.predictions = tf.argmax(xWb, 1, name="predictions")
            loss = tf.nn.softmax_cross_entropy_with_logits(xWb, self.input_labels)
            self.loss = tf.reduce_mean(loss) + l2_reg_lambda * l2_loss
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_labels, 1))
            self.accuracys = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracys")

    def fit(self, datas, epochs, keep_prob=0.5):
        import datetime
        logging.basicConfig(filename="{}.log".format(self.model_name), format="%(message)s", filemode="a",
                            level=logging.INFO)
        checkpoint_dir = os.path.abspath(os.path.join(self.model_name, "checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        with tf.Session() as sess:
            self.sess = sess
            optimizer = tf.train.AdamOptimizer(self.lr)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            train_op = optimizer.apply_gradients(grads_and_vars)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)
            sess.run(tf.initialize_all_variables())

            def train_step(x_batch, y_batch, z_batch):
                feed_dict = {self.input_word_id: x_batch, self.input_labels: y_batch, self.input_characters_id: z_batch, self.dropout_keep_prob: keep_prob}
                _, loss, accuracys = sess.run([train_op, self.loss, self.accuracys], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                logging.info("{}: loss: {}, acc: {}".format(time_str, loss, accuracys))
            for i in range(epochs):
                datas.shuffle()
                for _ in range(datas.batch_nums):
                    x_batch, y_batch, z_batch = datas.next_batch()
                    train_step(x_batch, y_batch, z_batch)
                saver.save(sess, checkpoint_prefix, global_step=i)  # 存储训练数据

    def eval(self, datas):
        def eval_step(x_batch, y_batch, z_batch):
            feed_dict = {self.input_word_id: x_batch, self.input_labels: y_batch, self.input_characters_id: z_batch, self.dropout_keep_prob: 1.0}
            accuracys, predictions = self.sess.run([self.accuracys, self.predictions], feed_dict)
            return accuracys, predictions

        eval_acc = []
        for _ in range(datas.batch_nums):
            x_batch, y_batch, z_batch = datas.next_batch()
            acc, pres = eval_step(x_batch=x_batch, y_batch=y_batch, z_batch=z_batch)
            eval_acc.append(acc)
        print np.mean(eval_acc)

    def load(self, checkpointfile):
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
            self.sess = sess
            saver = tf.train.import_meta_graph("{}.meta".format(checkpointfile))
            saver.restore(sess, checkpointfile)
            self.input_word_id = graph.get_operation_by_name("input_words_id").outputs[0]
            self.input_characters_id = graph.get_operation_by_name("input_characters_id").outputs[0]
            self.input_labels = graph.get_operation_by_name("input_labels").outputs[0]
            self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            self.accuracys = graph.get_operation_by_name("accuracy/accuracys").outputs[0]
            self.scores = graph.get_operation_by_name("output/scores").outputs[0]

    def predict(self, input_x, input_z):
        feed_dict = {self.input_word_id: input_x, self.input_characters_id:input_z, self.dropout_keep_prob: 1.0}
        predictions, scores = self.sess.run([self.predictions, self.scores], feed_dict=feed_dict)
        return predictions, scores