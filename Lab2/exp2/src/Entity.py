import re
import nltk
import string
import numpy as np
# 导入数据集预处理、特征工程和模型训练所需的库
from sklearn import model_selection, metrics
import pandas
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from keras.utils import to_categorical

np.set_printoptions(threshold=np.inf)

MAX_SEQUENCE_LENGTH = 36
EMBEDDING_DIM = 300

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

texts = []          # 训练文档的句子集合
labels = []         # 训练文档的关系集合
test_texts = []     # 测试文档的句子集合
pre_answer = []     # 预测的测试集的关系集合
entityextract_text = [] # 测试文档的句子集合,未去除标点符号,为了句子完整性，方便抽出实体
text_entity = []   # 记录文本中所有名词实体
total_answer = []   # 总的答案


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
        # remove = str.maketrans(string.punctuation, replacement)
        remove = str.maketrans(string.punctuation, replacement)
        sentence = sentence.translate(remove)
        tokens = sentence.split()
        sentence = ' '.join(tokens)
        texts.append(sentence)

        label = Relation2Num[relationship]
        # label = relationship
        labels.append(label)

        line = fd.readline()
        relation = fd.readline()
    fd.close()


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
        remove = str.maketrans(string.punctuation, replacement)
        sentence = sentence.translate(remove)
        tokens = sentence.split()
        sentence = ' '.join(tokens)  # 这一步有必要吗？
        test_texts.append(sentence)
        line = fd.readline()
    fd.close()


train_txt = "./train.txt"
test_txt = "./test.txt"
pretreatment(train_txt)
pretreatment_test(test_txt)

# 创建一个dataframe，列名为text和label
trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels

testDF = pandas.DataFrame()
testDF['text'] = test_texts

totalDF = pandas.DataFrame()
totalDF['text'] = texts + test_texts
# print(trainDF)
# os.system("pause")
# 将数据集分为训练集和验证集
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], train_size=0.9,
                                                                      test_size=0.1)

test_x = testDF['text']

# 加载预先训练好的词嵌入向量
embeddings_index = {}
for i, line in enumerate(open('./wiki-news-300d-1M/wiki-news-300d-1M.vec', encoding='utf-8')):
    values = line.split()
    embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

# 创建一个分词器
token = text.Tokenizer()
token.fit_on_texts(totalDF['text'])
word_index = token.word_index  # 获得单词:单词下标的字典
# print(word_index)

# 将文本转换为分词序列，并填充它们保证得到相同长度的向量
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=MAX_SEQUENCE_LENGTH)
# print(train_x[0], token.texts_to_sequences(train_x)[0])
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=MAX_SEQUENCE_LENGTH)
test_seq_x = sequence.pad_sequences(token.texts_to_sequences(test_x), maxlen=MAX_SEQUENCE_LENGTH)

# print(train_seq_x)

# 创建分词嵌入映射
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


def train_model(classifier, feature_vector_train, label, feature_vector_valid, test_vector):
    global pre_answer
    # fit the training dataset on the classifier
    classifier.fit(
                    feature_vector_train,
                    to_categorical(label, 10),
                    epochs=200,
                    batch_size=512,
                    validation_data=(feature_vector_valid, to_categorical(valid_y, 10))
    )
    predictions = classifier.predict(feature_vector_valid)
    predictions_test = classifier.predict(test_vector)
    # print("sadsgdfgfdgs: ", predictions)
    # os.system("pause")
    predictions = predictions.argmax(axis=-1)
    predictions_test = predictions_test.argmax(axis=-1)
    for i in range(len(predictions_test)):
        pre_answer.append(Num2Relation[predictions_test[i]])
    print("sad: ", predictions)
    return metrics.accuracy_score(predictions, valid_y)


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
    model.summary()
    # os.system("pause")
    return model


def create_cnn():
    # Add an Input Layer
    input_layer = layers.Input((MAX_SEQUENCE_LENGTH, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(256, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(128, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(10, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model


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


def ExtractENTITY():
    text_rank = 0
    for sentence in entityextract_text:
        text_rank += 1
        tokenized = nltk.word_tokenize(sentence) # 分词
        # pprint.pprint(tokenized)
        tagged = nltk.pos_tag(tokenized)  # 词性标注
        # pprint.pprint(tagged)
        chunked = nltk.ne_chunk(tagged)  # 命名实体识别
        '''
        NN 名词   year, home, costs, time, education
        NNS 名词复数 undergraduates scotches 
        NNP 专有名词 Alison, Africa, April, Washington
        NNPS 专有名词复数 Americans Americas Amharas Amityvilles
        '''
        # pprint.pprint(chunked)  # <class 'nltk.tree.Tree'>
        # print(chunked.draw())
        index = 0   # 标注在句子中的第几个单词
        nouns = []  # 记录所有非专有名词和非专有名词复数
        for tree in chunked:
            # print(type(tree))   # 非专有名词为'tuple',是专有名词的为“tree”
            # os.system("pause")
            index += 1
            if not hasattr(tree, 'label'):  # 非专有名词, 默认抽取句子中的非专有名词
                if tree[1] == 'NN' or tree[1] == 'NNS':
                    nouns.append([index, tree[0]])

        noun = []    # 记录一个单独的名词
        all_entity = []
        entity_term = []
        start = True      # 标记一个实体的开始处
        for j in range(1, len(nouns)):
            i = j - 1   # 判断两个名词在句子中出现的次序
            if start:
                noun.append(nouns[i][1])
                start = False
            if nouns[j][0] - nouns[i][0] == 1:
                noun.append(nouns[j][1])
                if j == len(nouns)-1:   # 最后一个元素, 且未与前面的名词组成连词, 则单独添加
                    # print(text_rank, noun)
                    phrase = ' '.join(noun)
                    all_entity.append(phrase)
                    noun = []
            else:
                start = True
                # print(text_rank, noun)
                phrase = ' '.join(noun)
                all_entity.append(phrase)
                noun = []
                if j == len(nouns)-1:   # 最后一个元素, 且未与前面的名词组成连词, 则单独添加
                    noun.append(nouns[j][1])
                    # print(text_rank, noun)
                    phrase = ' '.join(noun)
                    all_entity.append(phrase)
                    noun = []
        if len(all_entity) == 1:       # 抽取失败情况
            all_entity.append('e2')
        if len(all_entity) == 0:
            all_entity.append('e1')
            all_entity.append('e2')
        # print(all_entity)

        for entity in all_entity:
            if len(entity.split()) == 1:
                entity_term.append(entity)
        if len(entity_term) < 2:
            for entity in all_entity:
                if len(entity.split()) > 1:
                    entity_term.append(entity)
        # print(entity_term)
        text_entity.append([text_rank, entity_term[0], entity_term[1]])
    print(text_entity)


def main():
    # os.system("pause")
    classifier1 = create_rnn_gru()
    classifier2 = create_cnn()
    classifier3 = create_bidirectional_rnn()
    classifier4 = create_rcnn()
    classifier5 = create_model_architecture()
    ExtractENTITY()
    accuracy = train_model(classifier2, train_seq_x, train_y, valid_seq_x, test_seq_x)
    for i in range(len(pre_answer)):
        string = pre_answer[i] + '&' + text_entity[i][1] + ',' + text_entity[i][2]
        total_answer.append(string)

    aim_txt = './aim.txt'   # 目标文件
    fp = open(aim_txt, 'w')
    ans_string = '\n'.join(total_answer)
    fp.write(ans_string)
    fp.close()
    print("cnn, accuracy: ", accuracy)
    print("ok!!!")


main()



