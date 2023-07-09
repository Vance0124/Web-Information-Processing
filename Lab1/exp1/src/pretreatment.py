# coding = utf-8
#@Time : 2020/11/21 20:46
#@Author : zyc
#@File : pretreatment.py
#@Software: PyCharm

import os
import re
import nltk
from nltk.corpus import stopwords
import nltk.stem
import string
import json
import numpy as np
import math
inve_table = {}     # 倒排表
# term_count = []     # 词项频率
doc_count = np.zeros(1000)      # 文档频率
terms = []          # 词项


def find_file(path):     # Path represents the relative path
    index = 0            # Index represents the subscript of a list element
    file_list = os.listdir(path)     # 获得当前目录下的所有文件(文件夹)
    for i in range(len(file_list)):
        file = file_list[index]
        relative_path = path + "/" + file  # relative path
        if os.path.isdir(relative_path):
            subfile = find_file(relative_path)  # deep search
            for sub_file in subfile:
                file_list.append(sub_file)
            del file_list[index]     # 删除非文件的列表元素，索引 index 不变
        elif os.path.isfile(relative_path):
            file_list[index] = relative_path
            index = index + 1
    return file_list


def pretreatment(filelist, target_path, filelist_path):             # 预处理
    find_content = "\n\n(.*)"
    # find_content_before_re = "\n\n(.*?)-----Original"       # 用 Original 区分回复(re)文件的前后
    find_subject = "subject: (.*?)\n"

    # trans_punctuation = {',': ' ', '.': ' ', '?': ' ', '!': ' ', ';': ' ', '(': ' ', ')': ' '}

    stem = nltk.stem.SnowballStemmer('english')  # 参数是选择的语言, 用于词根化

    index = 1       # index 表示文档编号

    string_replace = string.punctuation + '0123456789'
    replacement = ''  # 用 ' ' 替换 string.punctuation 中的每个符号
    for i in range(len(string_replace)):
        replacement = replacement + ' '
    # replacement = ''            # 用 ' ' 替换 string.punctuation 中的每个符号
    # for i in range(len(string.punctuation)):
    #     replacement = replacement + ' '

    # 打印所有文档路径
    document_map = dict(zip(range(len(filelist)), filelist))
    f = open(filelist_path, 'w')
    handler = json.dumps(document_map)
    f.write(handler)
    f.close()

    # with open(target_path, 'a+') as wfp:
    with open(target_path, 'w') as wfp:
        for file in filelist:
            fp = open(file, 'r', errors='ignore')
            txt = fp.read()
            txt = txt.lower()     # 将文本全转化为小写
            fp.close()

            subject = re.findall(find_subject, txt, re.M | re.I)[0]      # 得到主题
            content = re.findall(find_content, txt, re.S)[0]      # 得到邮件正文内容

            # string.punctuation中包含英文的标点，我们将其放在待去除变量remove中
            # 函数需要三个参数，前两个表示字符的映射，我们是不需要的。
            remove = str.maketrans(string_replace, replacement)     # 去除标点符号
            content = content.translate(remove)
            subject = subject.translate(remove)

            # 由于使用nltk库分词会大大降低对51w个文档的处理效率，改为使用split分词
            # tokens = nltk.word_tokenize(content)        # 分词 spilt
            # sub_tokens = nltk.word_tokenize(subject)
            tokens = content.split()
            sub_tokens = subject.split()

            # tokens_nosw = [word for word in tokens if word not in stopwords.words('english')]   # 去除停用词
            # sub_tokens_nosw = [word for word in sub_tokens if word not in stopwords.words('english')]

            # tokens_stem = [stem.stem(ws) for ws in tokens_nosw]     # 词根化
            # sub_tokens_stem = [stem.stem(ws) for ws in sub_tokens_nosw]
            tokens_stem = [stem.stem(ws) for ws in tokens]  # 词根化
            sub_tokens_stem = [stem.stem(ws) for ws in sub_tokens]

            content_fin = ' '.join(tokens_stem)                     # 用空格拼接内容
            subject_fin = ' '.join(sub_tokens_stem)

            mail = subject_fin + ' ' + content_fin + '\n\n' + '-----Document demarcation-----' + '\n\n'   # 以 '-----Document demarcation-----' 划分每个文档
            wfp.write(mail)             # 写入预处理文档
            print(index)
            index = index + 1           # 处理下一个文档


def get_top_1k_words(txt_1, txt_2):
    global terms

    wordlist = {}

    fd = open(txt_1, 'r')
    line = fd.readline()
    # f1.close()
    # contents = re.sub("-----Document demarcation-----", "", contents)
    while line:
        if line != "-----Document demarcation-----\n":
            words = line.split()
            for word in words:
                # p = False
                # for i in range(len(wordlist)):
                #     if word == wordlist[i][0]:
                #         wordlist[i][1] += 1
                #         p = True
                #         break
                # if not p:
                #     wordlist.append([word, 1])
                if word in wordlist:
                    wordlist[word] += 1
                else:
                    wordlist[word] = 1
        line = fd.readline()
    fd.close()
    # wordlist.sort(key=lambda x: x[1], reverse=True)
    # 范围改为1000
    f2 = open(txt_2, 'w')
    for i in range(1000):
        # terms.append(wordlist[i][0])
        word_top = max(wordlist, key=wordlist.get)      # 用 max 取出字典中的 top1000 的键
        terms.append(word_top)
        f2.write(word_top + '\n')
        wordlist[word_top] = -1
    f2.close()


def create_inve_table(txt_1, out_doc_count_txt):
    global terms,  inve_table   # terms 为 top1000 词项, inve_table 为 倒排表(字典)
    fd = open(txt_1, 'r')
    line = fd.readline()
    doc_id = 0                  # 文档编号
    count = np.zeros(1000)      # 统计 top1000 的词项在每篇文档中的出现次数
    flag = np.zeros(1000)         # 用于标记1000个词是否已经在文档中出现过
    wfd = open(out_doc_count_txt, 'w')
    while line:
        if line != "-----Document demarcation-----\n":
            words = line.split()
            for word in words:
                if word in terms:
                    count[terms.index(word)] += 1   # 出现次数加 1
                    if word in inve_table:
                        if flag[terms.index(word)] != 1:    # 若第一次在这篇文档中出现, 则添加该文档编号
                            inve_table[word].append(doc_id)
                    else:
                        inve_table[word] = [doc_id]         # 若第一次出现该词, 则在字典中添加一个新的键值对
                    flag[terms.index(word)] = 1
        else:
            str_term_count = ''
            for i in range(1000):
                if count[i] > 0:
                    str_term_count = str_term_count + str(i + 1) + ' ' + str(count[i]) + ' '
            str_term_count = str_term_count + '\n'

            wfd.write(str_term_count)       # 将每篇文档的词项频率写入

            count = np.zeros(1000)
            doc_id += 1
            flag = np.zeros(1000)           # 重置 flag 列表, 重新标记一封新的邮件中是否出现top1000词项
        line = fd.readline()
    wfd.close()
    fd.close()


# 将倒排表写入指定文件中, 同时生成文档频率 doc_count, 并同样写入指定文件中
def write_index_file(path1, path2):
    global doc_count, inve_table
    f = open(path1, 'w')
    fd = open(path2, 'w')
    for word, docIDS in inve_table.items():
        doc_count[terms.index(word)] = len(docIDS)
        indexing = word + '\t' + str(len(docIDS)) + '\t'
        for docID in docIDS:
            indexing += (' ' + str(docID))
        f.write(indexing+'\n')
    f.close()
    map_doc_count = map(str, doc_count)
    str_doc_count = ' '.join(map_doc_count)
    fd.write(str_doc_count)
    fd.close()


# def get_doc_count(path):
#     global doc_count
#     fd = open(path, 'r')
#     line = fd.readline()
#     doc_count = line.split()
#     fd.close()


def doc_tf_idf(n, tc_path, output_txt):  # n 表示文档总数
    global doc_count
    fd = open(tc_path, 'r')
    line = fd.readline()
    wfd = open(output_txt, 'w')
    while line:
        doc_term = line.split()
        nums = int(len(doc_term)/2)
        element = ''
        for term_id in range(nums):           # 只记录了词项频率非0的词项, 因此 tf 不可能为 0
            if doc_count[int(doc_term[2*term_id]) - 1] != n:  # 等于 n 的话, tf_idf 的值就为 0, 不需要存储
                tf_idf = round((1 + math.log(float(doc_term[2*term_id+1]), 10)) * math.log(n/doc_count[int(doc_term[2*term_id]) - 1], 10), 6)
                element = element + str(doc_term[2*term_id]) + ' ' + str(tf_idf) + ' '
        element = element + '\n'
        wfd.write(element)
        line = fd.readline()
    wfd.close()
    fd.close()


filelist_path = '../output/doc_map.txt'
target_1 = '../output/pretreatment.txt'
target_2 = '../output/top1kwords.txt'
inve_txt = '../output/inverse_index.txt'
doc_count_txt = '../output/doc_count.txt'
term_count_txt = '../output/term_count.txt'
tf_idf_txt = '../output/tf_idf.txt'

dataset = '../dataset/enron_mail_20150507'
# test = './test'
filelist = find_file(dataset)
pretreatment(filelist, target_1, filelist_path)
get_top_1k_words(target_1, target_2)
create_inve_table(target_1, term_count_txt)
write_index_file(inve_txt, doc_count_txt)
N = 517401
# get_doc_count(doc_count_txt)
doc_tf_idf(N, term_count_txt, tf_idf_txt)
print("\nok!!!")