# -*- coding: utf-8 -*-

import re
import json
import nltk
import nltk.stem
import string

# 考虑到查询操作需要多次执行，把 index 作为全局变量，否则 search_index() 需要传递
# index 参数, document_map 同
inverse_index = {}      # 索引字典
# document_map = {}


# 从文件中读取索引创建索引字典
def create_index(file_name):
    global inverse_index
    # inverse_index = {}
    f = open(file_name, 'r')
    line = f.readline().rstrip()
    while line:
        line = re.split(r'\s+', line)
        # print(line)
        inverse_index[line[0]] = set()
        for id in line[2:]:
            inverse_index[line[0]].add(id)
        line = f.readline().rstrip()
    return inverse_index


# 从文件中读取 ID 表创建 document_map
def create_map(file_name):
    f = open(file_name, 'r')
    handler = f.read()
    doc_map = json.loads(handler)
    f.close()
    return doc_map


# 输出结果函数
def print_result(ids, doc_map):
    if len(ids) == 0:
        print("\nSorry,can not find the message!\n")
    else:
        result = ''
        i = 1
        for id in ids:
            result += str(i) + ': ' + doc_map[id] + '\n'
            i += 1
        print('\n', result)


# 在索引字典中搜索 keywords
# 因为输入查询的格式已经约定所以使用了下标的方法获得数据
def search_index(keywords, doc_map, n):
    global inverse_index
    opcode = []  # 用opcode列表充当操作码栈
    operand = []  # 用operand列表充当操作数栈
    full = set()
    for id in range(n):  # 构建全集
        full.add(str(id))
    stem = nltk.stem.SnowballStemmer('english')
    keywords = [stem.stem(ws) for ws in keywords]
    for token in keywords:
        if token == '(' or token == 'not':
            opcode.append(token)
        elif token == 'and':
            if len(opcode) and opcode[len(opcode) - 1] == 'and':
                preval = operand.pop()
                pre2val = operand.pop()
                operand.append(preval & pre2val)
            else:
                opcode.append(token)
        elif token == 'or':
            if len(opcode) and opcode[len(opcode) - 1] == 'and':
                opcode.pop()
                preval = operand.pop()
                pre2val = operand.pop()
                operand.append(preval & pre2val)
                opcode.append(token)
            elif len(opcode) and opcode[len(opcode) - 1] == 'or':
                preval = operand.pop()
                pre2val = operand.pop()
                operand.append(preval | pre2val)
            else:
                opcode.append(token)
        elif token == ')':
            preop = opcode.pop()
            while preop != '(':
                preval = operand.pop()
                pre2val = operand.pop()
                if preop == 'and':
                    operand.append(preval & pre2val)
                else:
                    operand.append(preval | pre2val)
                preop = opcode.pop()
            if len(opcode) and opcode[len(opcode) - 1] == 'not':
                opcode.pop()
                preval = operand.pop()
                operand.append(full - preval)
        else:
            if len(opcode) and opcode[len(opcode) - 1] == 'not':
                opcode.pop()
                operand.append(full - inverse_index.get(token, set()))
            else:
                operand.append(inverse_index.get(token, set()))
    while len(opcode):
        preop = opcode.pop()
        preval = operand.pop()
        pre2val = operand.pop()
        if preop == 'and':
            operand.append(preval & pre2val)
        else:
            operand.append(preval | pre2val)
    ids = operand.pop()
    print_result(ids, doc_map)


def bool_search():
    inve_txt = '../output/inverse_index.txt'
    filelist_path = '../output/doc_map.txt'
    create_index(inve_txt)	# 创建索引表
    doc_map = create_map(filelist_path)	# 创建文档路径字典
    total = len(doc_map)	# 文档(邮件)总数
    while True:
        query = input('请输入布尔查询的内容: ')
        query = query.lower()
        # 识别替换括号，去除两侧空白字符
        query = re.sub('\(', ' ( ', query)
        query = re.sub('\)', ' ) ', query).strip()
        # 切词
        keywords = re.split(r'\s+', query)
        if keywords:
            search_index(keywords, doc_map, total)
        else:
            print("Input does not contain keywords!")


if __name__ == '__main__':
    bool_search()
