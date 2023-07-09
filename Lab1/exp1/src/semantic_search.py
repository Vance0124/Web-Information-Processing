#-*- coding = utf-8 -*-
#@Time : 2020/11/22 17:44
#@Author : zyc
#@File : semantic_search.py
#@Software: PyCharm

import os
import re
import nltk
from nltk.corpus import stopwords
import nltk.stem
import string
import numpy as np
import math
import json
# import datetime

# inve_table = {}     # 倒排表
# term_counts = []    # 词项频率, 以文档 doc 的 id 作为索引(index), 方便相关度的计算
# doc_counts = []     # 文档频率, 以 词项的 id 为索引(index)
# terms = []          # 词项, 假设已经知道 top1000 的词项
dtifs = []          # 存储 tf_idf 矩阵, dtifs 中共有 文档ID 个子列表, 子列表中采用二元组存储,即(词项ID, tf_idf)列表, 以压缩稀疏矩阵


# 从文件中读取 ID 表创建 document_map
def create_map(file_name):
    f = open(file_name, 'r')
    handler = f.read()
    doc_map = json.loads(handler)
    f.close()
    return doc_map


# 创建文档频率 doc_counts
def get_doc_count(path):
    fd = open(path, 'r')
    line = fd.readline()
    doc_count = line.split()
    fd.close()
    return doc_count


# 创建 top1000 的词项
def get_terms(path):
    fd = open(path, 'r')
    line = fd.read()
    terms = line.split()
    fd.close()
    return terms


# 读取 tf_idf 文档
def get_tf_idf(path):
    global dtifs
    # dtifs = []
    fd = open(path, 'r')
    line = fd.readline()
    # index = 1
    while line:
        # print(index, line)
        # index += 1
        if line != '\n':
            doc_tf = line.split()
            dtifs.append(doc_tf)
        else:
            doc_tf = []
            dtifs.append(doc_tf)
        line = fd.readline()
    fd.close()
    # return dtifs


def semantic_search(query, n, terms, doc_counts):     # query 为查询词项的列表, n 表示文档总数
    # global terms, doc_counts
    global dtifs
    relevancy = []  # 用于记录文档的相关度
    doc = []
    relevancy_top10 = []    # 用于记录最相关的 10 篇文档的相关度
    vec_q = np.zeros(1000)
    for term in query:
        if term in terms:
            vec_q[terms.index(term)] += 1.0
    for i in range(len(vec_q)):
        if vec_q[i] != 0:
            vec_q[i] = round((1 + math.log(vec_q[i], 10)) * math.log(n/float(doc_counts[i]), 10), 10)

    vec_q_np = np.array(vec_q)

    for tds in dtifs:
        vec_doc = np.zeros(1000)
        nums = int(len(tds)/2)
        for term_id in range(nums):
            vec_doc[int(tds[2 * term_id]) - 1] = float(tds[2 * term_id + 1])

        vec_doc_np = np.array(vec_doc)

        value = np.sqrt((vec_q_np * vec_q_np).sum() * (vec_doc_np * vec_doc_np).sum())
        if value != 0:
            cosine = ((vec_q_np * vec_doc_np).sum())/value  # 求相关度
        else:
            cosine = 0
        relevancy.append(cosine)                       # 求得相关度的列表
    for i in range(10):
        index = relevancy.index(max(relevancy))
        doc.append(index)
        relevancy_top10.append(relevancy[index])
        relevancy[index] = -1       # -1 表示已经用完,不对后面的分析产生影响
    return doc, relevancy_top10


def query_process(query):
    # 对查询做预处理
    query = query.lower()
    string_replace = string.punctuation + '0123456789'
    replacement = ''  # 用 ' ' 替换 string_replace 中的每个符号
    for i in range(len(string_replace)):
        replacement = replacement + ' '
    remove = str.maketrans(string_replace, replacement)  # 去除标点符号
    query = query.translate(remove)
    words = query.split()
    # words = [word for word in words if word not in stopwords.words('english')]  # 去除停用词
    stem = nltk.stem.SnowballStemmer('english')  # 参数是选择的语言, 用于词根化
    tokens_stem = [stem.stem(ws) for ws in words]  # 词根化
    return tokens_stem


def semantic():
    filelist_path = '../output/doc_map.txt'
    doc_count_txt = '../output/doc_count.txt'
    tf_idf_txt = '../output/tf_idf.txt'
    terms_txt = '../output/top1kwords.txt'

    print("Please wait a few minutes for loading...")

    doc_map = create_map(filelist_path)           # 创建映射
    doc_count = get_doc_count(doc_count_txt)        # 创建文档频率 doc_counts
    terms = get_terms(terms_txt)                # 创建 top1000 的词项
    get_tf_idf(tf_idf_txt)                      # 读取 tf_idf 文档
    n = len(doc_map)  # 文档总数

    while True:
        query = input('请输入语义查询的内容:')
        # start = datetime.datetime.now()
        tokens_stem = query_process(query)
        doc, relevancy_top10 = semantic_search(tokens_stem, n, terms, doc_count)
        print('The top 10 most relevant documents:')
        for i in range(len(doc)):
            print(i+1, ':', doc_map[str(doc[i])], "\t\t relevancy: ", relevancy_top10[i])
        # end = datetime.datetime.now()
        # print("runtime: ", end - start)


if __name__ == '__main__':
    semantic()
