# coding=utf-8
"""
命名实体识别
2017-03-17 张洛阳 南京
"""
from model import *
from datasets import *
import json
import argparse
from word2vec import word2vec

def parse_args():
    parser = argparse.ArgumentParser(description="neural network for named entity recongnition")
    parser.add_argument('--model', help='neural networl model, default is mlp_model', default='mlp_truned_character_model', type=str)
    parser.add_argument('--embedding_size', help='word embedding size, default=100', default=100, type=int)
    parser.add_argument('--character_length', help='character length, dafault=5', default=5, type=int)
    parser.add_argument('--character_embedding', help='character embedding size, dafault=50', default=50, type=int)
    parser.add_argument('--context_size' ,help='windows size for neural network, must be a odd number, defalut=9', default=9, type=int)
    parser.add_argument('--filter_length', help='filter length', default=3, type=int)
    parser.add_argument('--filter_nums', help='numbers of convolution kernel', default=100, type=int)
    parser.add_argument('--hidden_layer' ,help='hidden layer neurals', default=100, type=int)
    parser.add_argument('--dropout_keep_prob', help='dropout keep probability, default=0.5', default=0.5, type=float)
    parser.add_argument('--batch_size', help='default=64', default=64, type=int)
    parser.add_argument('--num_epochs', help='default=20', default=100, type=int)
    parser.add_argument('--train_data', help='datasets train file', default='data/ner_train_ltp', type=str)
    parser.add_argument('--test_data', help='datasets test file', default='data/ner_test_ltp', type=str)
    parser.add_argument('--word2vec', help='pretrain word2vec model', default='data/vectors.bin', type=str)
    parser.add_argument('--data_exists', help='datas has already processed, default=0', default=0, type=int)
    parser.add_argument('--train', help='train, test or demo, default=train', default='train', type=str)
    return parser.parse_args()


def train_simple_mlp_model(args):
    embedding_size = args.embedding_size
    context_size = args.context_size
    batch_size = args.batch_size
    dropout_keep_prob = args.dropout_keep_prob
    train_data = args.train_data
    test_data = args.test_data
    num_epochs = args.num_epochs
    data_exists = args.data_exists
    datas = DataSet(train_data, test_data, batch_size, context_size, embedding_size, data_exists)
    vocab_size = datas.vocab_size
    num_classes = datas.labels_size
    hidden_layer = args.hidden_layer
    params = {"train_data":train_data, "test_data":test_data, "embedding_size": embedding_size, 'context_size':context_size, 'vocab_size':vocab_size, "dropout_keep_prob": dropout_keep_prob, "num_classes": num_classes,
              "batch_size": batch_size, "num_epochs": num_epochs, "model": args.model, 'hidden_layer':hidden_layer}
    if not os.path.exists("conf"):
        os.mkdir("conf")
    json.dump(params, open("conf/{}.json".format(args.model), "w"))
    # batch_size=None
    model = eval(args.model)(context_size, embedding_size, vocab_size, num_classes, None, hidden_layer)
    model.fit(datas, num_epochs)


def test_simple_mlp_model(args):
    params = json.load(open("conf/{}.json".format(args.model), "r"))
    embedding_size = params['embedding_size']
    context_size = params['context_size']
    vocab_size = params['vocab_size']
    dropout_keep_prob = params['dropout_keep_prob']
    num_classes = params['num_classes']
    batch_size = 1
    num_epochs = params['num_epochs']
    hidden_layer = params['hidden_layer']
    test_data = params['test_data']
    batch_size = 1
    datas = DataSetEval(test_data, batch_size, context_size, embedding_size)
    checkpoint_dir = os.path.abspath(os.path.join("{}".format(args.model), "checkpoints"))
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    model = eval(args.model)(context_size, embedding_size, vocab_size, num_classes, None, hidden_layer)
    model.load(checkpoint_file)
    model.eval(datas)


def demo_simple_mlp_model(args):
    params = json.load(open("conf/{}.json".format(args.model), "r"))
    embedding_size = params['embedding_size']
    context_size = params['context_size']
    vocab_size = params['vocab_size']
    dropout_keep_prob = params['dropout_keep_prob']
    num_classes = params['num_classes']
    batch_size = 1
    num_epochs = params['num_epochs']
    hidden_layer = params['hidden_layer']
    test_data = params['test_data']
    batch_size = 1
    datas = DataSetConvert(batch_size, context_size, embedding_size)
    checkpoint_dir = os.path.abspath(os.path.join("{}".format(args.model), "checkpoints"))
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    model = eval(args.model)(context_size, embedding_size, vocab_size, num_classes, None, hidden_layer)
    model.load(checkpoint_file)
    words_labels = ['与/O', '福安/B-ni', '药业/E-ni', '一同/O', '上市/O', '的/O', '佳士/B-ni', '科技/E-ni', '昨日/O', '表现/O', '同样/O', '不/O', '佳/O']
    words_id, labels_id = datas.convert(words_labels)
    predict, _ = model.predict(words_id)
    predict = map(lambda p: datas.all_labels[p], predict)
    print predict


def train_mlp_truned_model(args):
    # embedding_size = args.embedding_size
    context_size = args.context_size
    batch_size = args.batch_size
    dropout_keep_prob = args.dropout_keep_prob
    train_data = args.train_data
    test_data = args.test_data
    num_epochs = args.num_epochs
    data_exists = args.data_exists
    w2c_model = word2vec(args.word2vec)
    w2c = w2c_model.word2vec
    embedding_size = w2c_model.embedding_size
    datas = DatasetTruned(train_data, test_data, batch_size, context_size, embedding_size, w2c, data_exists)
    vocab_size = datas.vocab_size
    num_classes = datas.labels_size
    word2vec_init = datas.word_vec
    hidden_layer = args.hidden_layer
    params = {"train_data": train_data, "test_data": test_data, "embedding_size": embedding_size,
              'context_size': context_size, 'vocab_size': vocab_size, "dropout_keep_prob": dropout_keep_prob,
              "num_classes": num_classes,
              "batch_size": batch_size, "num_epochs": num_epochs, "model": args.model, 'hidden_layer': hidden_layer}
    if not os.path.exists("conf"):
        os.mkdir("conf")
    json.dump(params, open("conf/{}.json".format(args.model), "w"))
    batch_size=None
    model = eval(args.model)(context_size, embedding_size, vocab_size, num_classes, batch_size, hidden_layer, word2vec_init)
    model.fit(datas, num_epochs)


def test_mlp_truned_model(args):
    params = json.load(open("conf/{}.json".format(args.model), "r"))
    embedding_size = params['embedding_size']
    context_size = params['context_size']
    vocab_size = params['vocab_size']
    dropout_keep_prob = params['dropout_keep_prob']
    num_classes = params['num_classes']
    batch_size = 1
    num_epochs = params['num_epochs']
    hidden_layer = params['hidden_layer']
    test_data = params['test_data']
    batch_size = 1
    datas = DataSetTrunedEval(test_data, batch_size, context_size, embedding_size)
    checkpoint_dir = os.path.abspath(os.path.join("{}".format(args.model), "checkpoints"))
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    model = eval(args.model)(context_size, embedding_size, vocab_size, num_classes, batch_size, hidden_layer, np.zeros((vocab_size, embedding_size), dtype="float32"))
    model.load(checkpoint_file)
    model.eval(datas)


def demo_mlp_truned_model(args):
    params = json.load(open("conf/{}.json".format(args.model), "r"))
    embedding_size = params['embedding_size']
    context_size = params['context_size']
    vocab_size = params['vocab_size']
    dropout_keep_prob = params['dropout_keep_prob']
    num_classes = params['num_classes']
    batch_size = 1
    num_epochs = params['num_epochs']
    hidden_layer = params['hidden_layer']
    test_data = params['test_data']
    batch_size = 1
    datas = DataSetTrunedConvert(batch_size, context_size, embedding_size)
    checkpoint_dir = os.path.abspath(os.path.join("{}".format(args.model), "checkpoints"))
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    model = eval(args.model)(context_size, embedding_size, vocab_size, num_classes, batch_size, hidden_layer, np.zeros((vocab_size, embedding_size), dtype="float32"))
    model.load(checkpoint_file)
    words_labels = ['与/O', '福安/B-ni', '药业/E-ni', '一同/O', '上市/O', '的/O', '佳士/B-ni', '科技/E-ni', '昨日/O', '表现/O', '同样/O',
                    '不/O', '佳/O']
    words_id, labels_id = datas.convert(words_labels)
    predict, _ = model.predict(words_id)
    predict = map(lambda p: datas.all_labels[p], predict)
    print predict


def train_mlp_truned_character_model(args):
    # embedding_size = args.embedding_size
    context_size = args.context_size
    batch_size = args.batch_size
    dropout_keep_prob = args.dropout_keep_prob
    train_data = args.train_data
    test_data = args.test_data
    num_epochs = args.num_epochs
    character_length = args.character_length
    character_embedding = args.character_embedding
    data_exists = args.data_exists
    filter_length = args.filter_length
    filter_nums = args.filter_nums
    w2c_model = word2vec(args.word2vec)
    w2c = w2c_model.word2vec
    embedding_size = w2c_model.embedding_size
    datas = DataSeTCharater(train_data, test_data, batch_size, context_size, embedding_size, w2c, character_length, exists=data_exists)
    vocab_size = datas.vocab_size
    num_classes = datas.labels_size
    character_size = datas.character_size
    word2vec_init = datas.word_vec
    hidden_layer = args.hidden_layer
    params = {"train_data": train_data, "test_data": test_data, "embedding_size": embedding_size,
              'context_size': context_size, 'vocab_size': vocab_size, "dropout_keep_prob": dropout_keep_prob,
              "num_classes": num_classes, 'character_length':character_length, 'character_embedding':character_embedding,
              'filter_length':filter_length, 'filter_nums':filter_nums,
              "batch_size": batch_size, "num_epochs": num_epochs, "model": args.model, 'hidden_layer': hidden_layer}
    if not os.path.exists("conf"):
        os.mkdir("conf")
    json.dump(params, open("conf/{}.json".format(args.model), "w"))
    batch_size = None
    model = eval(args.model)(context_size, embedding_size, vocab_size, num_classes, batch_size, hidden_layer, word2vec_init, character_length, character_size, character_embedding,
                 filter_nums, filter_length)
    model.fit(datas, num_epochs)


def test_mlp_truned_character_model(args):
    params = json.load(open("conf/{}.json".format(args.model), "r"))
    embedding_size = params['embedding_size']
    context_size = params['context_size']
    vocab_size = params['vocab_size']
    num_classes = params['num_classes']
    dropout_keep_prob = params['dropout_keep_prob']
    num_classes = params['num_classes']
    batch_size = 1
    num_epochs = params['num_epochs']
    hidden_layer = params['hidden_layer']
    character_length = params['character_length']
    test_data = params['test_data']
    character_embedding = params['character_embedding']
    filter_nums = params['filter_nums']
    filter_length = params['filter_length']
    batch_size = 1
    datas = DataSeTCharaterEval(test_data, context_size, embedding_size, character_length, batch_size)
    checkpoint_dir = os.path.abspath(os.path.join("{}".format(args.model), "checkpoints"))
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    character_size = datas.character_size
    model = eval(args.model)(context_size,embedding_size, vocab_size, num_classes, batch_size, hidden_layer, np.zeros((vocab_size, embedding_size), dtype="float32"),
                     character_length, character_size, character_embedding,filter_nums, filter_length)
    model.load(checkpoint_file)
    model.eval(datas)


def demo_mlp_truned_character_model(args):
    params = json.load(open("conf/{}.json".format(args.model), "r"))
    embedding_size = params['embedding_size']
    context_size = params['context_size']
    vocab_size = params['vocab_size']
    num_classes = params['num_classes']
    dropout_keep_prob = params['dropout_keep_prob']
    num_classes = params['num_classes']
    batch_size = 1
    num_epochs = params['num_epochs']
    hidden_layer = params['hidden_layer']
    character_length = params['character_length']
    test_data = params['test_data']
    character_embedding = params['character_embedding']
    filter_nums = params['filter_nums']
    filter_length = params['filter_length']
    batch_size = 1
    datas = DataSeTCharaterConvert(context_size, embedding_size, character_length)
    character_size = datas.character_size
    checkpoint_dir = os.path.abspath(os.path.join("{}".format(args.model), "checkpoints"))
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    model = eval(args.model)(context_size, embedding_size, vocab_size, num_classes, batch_size, hidden_layer,
                             np.zeros((vocab_size, embedding_size), dtype="float32"),
                             character_length, character_size, character_embedding, filter_nums, filter_length)
    model.load(checkpoint_file)
    words_labels = ['与/O', '福安/B-ni', '药业/E-ni', '一同/O', '上市/O', '的/O', '佳士/B-ni', '科技/E-ni', '昨日/O', '表现/O', '同样/O',
                    '不/O', '佳/O']
    words_id, labels_id, characters_id = datas.convert(words_labels)
    predict, _ = model.predict(words_id, characters_id)
    predict = map(lambda p: datas.all_labels[p], predict)
    print predict

def main(args):
    model_name = args.model
    train_or_test = args.train
    if train_or_test == "train":
        if model_name == "mlp_model":
            train_simple_mlp_model(args)
        if model_name == "mlp_model_truned":
            train_mlp_truned_model(args)
        if model_name == "mlp_truned_character_model":
            train_mlp_truned_character_model(args)
    elif train_or_test == "test":
        if model_name == "mlp_model":
            test_simple_mlp_model(args)
        if model_name == "mlp_model_truned":
            test_mlp_truned_model(args)
        if model_name == "mlp_truned_character_model":
            test_mlp_truned_character_model(args)
    elif train_or_test == "demo":
        if model_name == "mlp_model":
            demo_simple_mlp_model(args)
        if model_name == "mlp_model_truned":
            demo_mlp_truned_model(args)
        if model_name == "mlp_truned_character_model":
            demo_mlp_truned_character_model(args)




if __name__ == "__main__":
    args = parse_args()
    main(args)