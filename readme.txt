author:zhangluoyang email:55058629@qq.com 2017-03-13 nanjing
neural network for named entity recognition
we use context word embedding of the target word as the features for neural network.
Also, wo contempt to use character level embedding to overcome unknow word in reality environment.
the context word embedding as the word level feature.
the character level embedding feature over convolution and max-pooling as the character level feature.
then, we concat the word level feature and the character level feature as the full features, input into a simple forword neural network.
environment dependence
python 2.7
esm 
tensorflow0.12
numpy
datasets: train.txt test.txt
word2vec bin file: vector.bin
format example:
易联众/nz, 、/w, 金亚/nz, 科技/n, 、/w, 兴源/ns, 过滤/v, 、/w, 宝莱特/nrf, 、/w, 天/qt, 舟/n, 文化/n, 等/udeng, 6股/mq, 跌停/nz
易联众/S-ni, 、/O, 金亚/B-ni, 科技/E-ni, 、/O, 兴源/B-ni, 过滤/E-ni, 、/O, 宝莱特/S-ni, 、/O, 天/B-ni, 舟/M-ni, 文化/E-ni, 等/O, 6股/O, 跌停/O

新疆/ns, 城建/n, (/w, 集团/nis, )/w, 股份有限公司/nis, 董事会/nis, 2011/m, 年/qt, 1/m, 月/n, 25/m, 日/b
新疆/B-ni, 城建/M-ni, (/M-ni, 集团/M-ni, )/M-ni, 股份有限公司/E-ni, 董事会/O, 2011/O, 年/O, 1/O, 月/O, 25/O, 日/O

each instance contains three lines, one for part-of-speech tagging, anthor one is labels, and the last one is split pan line.
I have realize three models, you can choice appropriate params on your own datas.
how to run:
	python main.py --model mlp_model --train train --num_epoch 20
	python main.py --model mlp_model --train test
	python main.py --model mlp_model --train demo

	python main.py --model mlp_model_truned --train train --num_epoch 20
	python main.py --model mlp_model_truned --train test
	python main.py --model mlp_model_truned --train demo

	python main.py --model mlp_truned_character_model --train train --num_epoch 20
	python main.py --model mlp_truned_character_model --train test
	python main.py --model mlp_truned_character_model --train demo


reference:
