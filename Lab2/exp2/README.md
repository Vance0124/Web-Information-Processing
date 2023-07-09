## lab2-信息抽取系统

<center>曾勇程  &emsp; PB18000268</center>

<center>张弛 &emsp; PB18000269</center>

### 实验运行环境

1. 硬件条件：一台PC机(联想拯救者Y7000)

2. 软件条件：

   系统Windows10

   语言python3.6(64位)

   编辑器pycharm2018.3.4



### 编辑运行方式

可在编辑器pycharm中或其他支持python的编辑器中运行(保证库已经`pip`下来)。

需要的库有：`tensorflow 2.4`，`sklearn`，`keras`，`panda`，`nltk`，`numpy`，`re`，`numpy`，`string`等库。

`python3.9`貌似还不支持`tensorflow 2.4`，因此我重新载了`python3.6`版本。

注：由于我下载的是CPU版本的`tensorflow 2.4`，因此运行时会有一些警告，不过这是正常的，一些警告如下：

~~~
2021-01-04 19:06:10.792535: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'cudart64_110.dll'; dlerror: cudart64_110.dll not found
2021-01-04 19:06:10.793844: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
~~~



### 文件含义

```
exp2/
|----src/
	|----Entity.py 				//关系抽取和实体识别代码
|----实验报告.pdf
|----README
```



### 关键函数说明

本次实验用到的库有：

~~~python
import re
import nltk
import string
import numpy as np
from sklearn import model_selection, metrics
import pandas
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from keras.utils import to_categorical
~~~

1. 首先，定义全局列表

~~~python
texts = []          # 训练文档的句子集合
labels = []         # 训练文档的关系集合
test_texts = []     # 测试文档的句子集合
pre_answer = []     # 预测的测试集的关系集合
entityextract_text = [] # 测试文档的句子集合,未去除标点符号,为了句子完整性，方便抽出实体
text_entity = []   # 记录文本中所有名词实体
total_answer = []   # 总的答案
~~~

2. 文档预处理

我们先对将要分类的十种类别进行编号，建立字典，并建立反向索引(根据分类类别找到对应的关系字符串)。

~~~python
# 关系对应标签
Relation2Num = {
    "Other": 0,
    "Cause-Effect": 1,
    "Component-Whole": 2,
    "Entity-Destination": 3,
    "Product-Producer": 4,
    "Entity-Origin": 5,
    "Member-Collection": 6,
    "Message-Topic": 7,
    "Content-Container": 8,
    "Instrument-Agency": 9
}
Num2Relation = {n: r for r, n in Relation2Num.items()}
~~~

随后预处理函数中，逐行读取文档，隔行依次为文本和对应关系，对于文本进行分词处理，对于对应关系进行标签转化。

~~~python
def pretreatment(train_txt):
    # 识别例句和关系类别
    get_sentence = "\"(.*)\""
    get_relationship = "(.*?)\("
    # 替换标点符号为空格
    replacement = ''  # 用 ' ' 替换 string.punctuation 中的每个符号
    for i in range(len(string.punctuation)):
        replacement = replacement + ' '
    # 打开文件
    fd = open(train_txt, 'r')
    # 逐行读取
    line = fd.readline()
    relation = fd.readline()
    while line:
        sentence = re.findall(get_sentence, line, re.S)[0]
        relationship = re.findall(get_relationship, relation, re.S)[0]
        # 分词
        sentence = sentence.lower()
        remove = str.maketrans(string.punctuation, replacement)	# 去除标点符号
        sentence = sentence.translate(remove)
        tokens = sentence.split()
        sentence = ' '.join(tokens)  
        texts.append(sentence)

        label = Relation2Num[relationship]		# 对应的关系标签
        labels.append(label)

        line = fd.readline()
        relation = fd.readline()
    fd.close()
~~~

上面是对训练集的处理，测试集中由于没有给明的关系，并且要整理句子用于实体识别，所以方法略而有不同，见下：

~~~python
def pretreatment_test(path):
    # 识别例句和关系类别
    get_sentence = "\"(.*)\""
    # 替换标点符号为空格
    replacement = ''  # 用 ' ' 替换 string.punctuation 中的每个符号
    for i in range(len(string.punctuation)):
        replacement = replacement + ' '
    # 打开文件
    fd = open(path, 'r')
    # 逐行读取
    line = fd.readline()
    while line:
        sentence = re.findall(get_sentence, line, re.S)[0]
        entityextract_text.append(sentence)     # 这个作为抽取测试的实体
        # 分词
        sentence = sentence.lower()
        remove = str.maketrans(string.punctuation, replacement) # 去除标点符号
        sentence = sentence.translate(remove)
        tokens = sentence.split()
        sentence = ' '.join(tokens)  
        test_texts.append(sentence)
        line = fd.readline()
    fd.close()
~~~

3. 划分训练集为训练集和验证集(9:1划分，**真正预测测试集时应该修改代码去掉验证集以扩大训练集！！！**)

~~~python
# 创建一个dataframe，列名为text和label
trainDF = pandas.DataFrame()
trainDF['text'] = texts		# 训练集句子集合
trainDF['label'] = labels	# 训练集关系集合

testDF = pandas.DataFrame()	# 测试集对应的dataframe
testDF['text'] = test_texts	# 测试集句子集合

totalDF = pandas.DataFrame()
totalDF['text'] = texts + test_texts
# 将数据集分为训练集和验证集
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], train_size=0.9, test_size=0.1)
test_x = testDF['text']
~~~

#### 关系抽取

3. 建立特征向量转化字典

这一步中我们先创建一个对词项转化为300维特征向量的字典，先使用分词器对出现词项进行统计和编号，获得出现的词项总数，再调用下载好的预训练模型“`wiki-news-300d-1M`”构建对于每个词项的对应字典项。

同时我们也进一步将文本转化为固定长度的分词序列形式，以便于导入训练模型，以及进行特征向量转化。

这里有两个预先定义的全局变量：

~~~python
MAX_SEQUENCE_LENGTH = 36	# 每个句子转化成的维数，通过不断训练找到的最适维数，过大会有很多0向量，过小句子表达不完整
EMBEDDING_DIM = 300		# 每个单词转化成的维数
~~~

下面对应单词和句子转化为向量的代码：

~~~python
# 加载预先训练好的词嵌入向量
embeddings_index = {}
for i, line in enumerate(open('./wiki-news-300d-1M/wiki-news-300d-1M.vec', encoding='utf-8')):
    values = line.split()
    embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')	# 将单词转化为300维向量

# 创建一个分词器
token = text.Tokenizer()
token.fit_on_texts(totalDF['text'])
word_index = token.word_index  # 获得单词总数

# 将文本转换为分词序列，并填充它们保证得到相同长度的向量，这里的句子向量维数为36维
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=MAX_SEQUENCE_LENGTH)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=MAX_SEQUENCE_LENGTH)
test_seq_x = sequence.pad_sequences(token.texts_to_sequences(test_x), maxlen=MAX_SEQUENCE_LENGTH)

# 创建分词嵌入映射
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
~~~

4. 导入神经网络

这一步中，我们将预处理好的模型导入各种机器学习和深度神经网络模型中，以找到训练效果最佳的模型。

以下是模型训练函数。

~~~python
def train_model(classifier, feature_vector_train, label, feature_vector_valid, test_vector):
    global pre_answer
    # 将对应数据导入分类器，设置训练参数进行训练
    classifier.fit(								# classifier为训练的模型
                    feature_vector_train,
                    to_categorical(label, 10),	# 将标签转化为one-hot(独热编码)的10维向量
                    epochs=160,			# 训练次数
                    batch_size=512,
                    validation_data=(feature_vector_valid, to_categorical(valid_y, 10))
    )
    # 得到验证集和测试集的预测结果
    predictions = classifier.predict(feature_vector_valid)	# 用于验证结果的准确性，得到验证集的正确率
    predictions_test = classifier.predict(test_vector)		# 测试集的预测答案
    predictions = predictions.argmax(axis=-1)				# 取每行概率最大的下标，即对应为预测的关系
    predictions_test = predictions_test.argmax(axis=-1)
    for i in range(len(predictions_test)):
        pre_answer.append(Num2Relation[predictions_test[i]])
    return metrics.accuracy_score(predictions, valid_y)
~~~

以下是各种类型的神经网络。

CNN模型（卷积神经网络）

~~~python
def create_cnn():
    # Add an Input Layer
    input_layer = layers.Input((MAX_SEQUENCE_LENGTH, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)(input_layer)		# 将单词的向量矩阵(embedding_matrix)作为权值矩阵，训练过程中，权值不变
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(256, 3, activation="relu")(embedding_layer)	# 256的意思是神经网络层数，采用的激活函数为relu函数——修正线性单元

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(128, activation="relu")(pooling_layer)	
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(10, activation="sigmoid")(output_layer1)	# 10对应输出的维数，由于标签是独热编码(10维)，故值为10

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy') # 损失函数为binary_crossentropy函数
	model.summary()
    
    return model
~~~

双向RNN模型

~~~python
def create_bidirectional_rnn():
    # Add an Input Layer
    input_layer = layers.Input((MAX_SEQUENCE_LENGTH, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = layers.Bidirectional(layers.GRU(128))(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(64, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(10, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model
~~~

RNN-GRU模型（使用门控递归单元GRU代替循环神经网络中的LSTM层）

~~~python
def create_rnn_gru():
    # Add an Input Layer
    input_layer = layers.Input((MAX_SEQUENCE_LENGTH, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the GRU Layer
    lstm_layer = layers.GRU(128)(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(64, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(10, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    
    return model
~~~

RCNN模型（循环神经网络）

~~~python
def create_rcnn():
    # Add an Input Layer
    input_layer = layers.Input((MAX_SEQUENCE_LENGTH, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the recurrent layer
    rnn_layer = layers.Bidirectional(layers.GRU(64, return_sequences=True))(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(128, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(64, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(10, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model
~~~

浅层神经网络

~~~python
def create_model_architecture():
    # create input layer
    input_layer = layers.Input((MAX_SEQUENCE_LENGTH, ), sparse=True)

    # create hidden layer
    hidden_layer = layers.Dense(128, activation="relu")(input_layer)

    # create output layer
    output_layer = layers.Dense(10, activation="sigmoid")(hidden_layer)

    classifier = models.Model(inputs=input_layer, outputs=output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier
~~~

经过训练发现，卷积神经网络不仅准确率高，而且运行速度最快，因此我们决定采用**卷积神经网络**。



#### 实体识别

实体识别的代码如下（讲解见注释）：

~~~python
def ExtractENTITY():
    text_rank = 0
    for sentence in entityextract_text:
        text_rank += 1
        tokenized = nltk.word_tokenize(sentence) # 分词
        tagged = nltk.pos_tag(tokenized)  # 词性标注
        chunked = nltk.ne_chunk(tagged)  # 命名实体识别,形成抽象语法树
        '''
        NN	: 名词   year, home, costs, time, education
        NNS	: 名词复数 undergraduates scotches 
        NNP	: 专有名词 Alison, Africa, April, Washington
        NNPS: 专有名词复数 Americans Americas Amharas Amityvilles
        '''
        index = 0   # 标注在句子中的第几个单词
        nouns = []  # 记录所有非专有名词和非专有名词复数
        for tree in chunked:	
            # print(type(tree))   # 非专有名词为'tuple',是专有名词的为“tree”
            index += 1
            if not hasattr(tree, 'label'):  # 非专有名词, 默认抽取句子中的非专有名词(含有label标签的为专有名词)
                if tree[1] == 'NN' or tree[1] == 'NNS':		# 为非专有名词
                    nouns.append([index, tree[0]])		# 将实体加入nouns中

        noun = []   # 记录一个单独的名词(有些名词是2个及以上的单词的整体，如high-definition broadcast由2个单词组成)
        all_entity = []
        entity_term = []
        start = True      # 标记一个实体的开始处
        for j in range(1, len(nouns)):
            i = j - 1   # 判断两个名词在句子中出现的次序
            if start:
                noun.append(nouns[i][1])	# 第一个单词直接加入noun
                start = False
            if nouns[j][0] - nouns[i][0] == 1:	# 默认在句子中相连出现的名词为一个整体
                noun.append(nouns[j][1])	# 如果出现次序只相差1则加入
                if j == len(nouns)-1:   # 最后一个元素, 且未与前面的名词组成连词, 则单独添加
                    phrase = ' '.join(noun)	# 合成一个名词
                    all_entity.append(phrase)
                    noun = []
            else:
                start = True		# 另一个名词出现处，重置 start 为true
                phrase = ' '.join(noun)
                all_entity.append(phrase)
                noun = []
                if j == len(nouns)-1:   # 最后一个元素, 且未与前面的名词组成连词, 则单独添加
                    noun.append(nouns[j][1])
                    phrase = ' '.join(noun)
                    all_entity.append(phrase)
                    noun = []
        if len(all_entity) == 1:       # 抽取失败情况(nltk有时抽取出来的名词数量小于2个)
            all_entity.append('e2')
        if len(all_entity) == 0:
            all_entity.append('e1')
            all_entity.append('e2')

        for entity in all_entity:
            if len(entity.split()) == 1:  # 通过观察，单个单词的名词出现的概率要大一些，因此优先单个单词的名词
                entity_term.append(entity)
        if len(entity_term) < 2:		# 如果单个单词的名词数量小于2，则将其他名词加入填补
            for entity in all_entity:
                if len(entity.split()) > 1:		# 为了避免重复
                    entity_term.append(entity)

        text_entity.append([text_rank, entity_term[0], entity_term[1]])
~~~

`main()`函数：

~~~python
def main():
    # os.system("pause")
    classifier1 = create_rnn_gru()				# 模型1
    classifier2 = create_cnn()					# 模型2
    classifier3 = create_bidirectional_rnn()	# 模型3
    classifier4 = create_rcnn()					# 模型4
    classifier5 = create_model_architecture()	# 模型5
    ExtractENTITY()								# 实体抽取
    accuracy = train_model(classifier2, train_seq_x, train_y, valid_seq_x, test_seq_x)# 开始训练，得出结果
    for i in range(len(pre_answer)):
        string = pre_answer[i] + '&' + text_entity[i][1] + ',' + text_entity[i][2]
        total_answer.append(string)

    aim_txt = './aim.txt'   # 目标文件
    fp = open(aim_txt, 'w')
    ans_string = '\n'.join(total_answer)
    fp.write(ans_string)
    fp.close()
    print("cnn, accuracy: ", accuracy)	# 准确率
    print("ok!!!")
~~~
