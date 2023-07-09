# -*- coding = utf-8 -*-
# @Time : 2020/11/29 23:19
# @Author : zyc
# @File : mail_retrieval_system.py
# @Software: PyCharm

import semantic_search as semantic
import bool_search
import re
import threading
# import datetime


def main():
    filelist_path = '../output/doc_map.txt'
    doc_count_txt = '../output/doc_count.txt'
    tf_idf_txt = '../output/tf_idf.txt'
    inve_txt = '../output/inverse_index.txt'   # global index, 从文件中读取索引创建索引字典用于查询
    terms_txt = '../output/top1kwords.txt'

    print("Please wait a few minutes for loading...")

    # start = datetime.datetime.now()

    t1 = threading.Thread(target=semantic.get_tf_idf, args=(tf_idf_txt,))  # 创建线程 1
    t1.start()  # 开始执行线程 1
    t2 = threading.Thread(target=bool_search.create_index, args=(inve_txt,))    # 创建线程 2
    t2.start()      # 开始执行线程 2

    doc_map = semantic.create_map(filelist_path)  # 创建映射
    doc_count = semantic.get_doc_count(doc_count_txt)  # 创建文档频率 doc_counts
    terms = semantic.get_terms(terms_txt)  # 创建 top1000 的词项

    t1.join()       # 等待线程 1 和 2 执行完
    t2.join()

    # end = datetime.datetime.now()
    # print("runtime: ", end - start)

    n = len(doc_map)  # 文档总数

    print("The retrieval method is as follows:")
    print("0) quit   1) bool search   2) semantic search")
    choose = input("Please enter the retrieval method:")
    while choose != '0':
        if choose == '1':
            query = input('请输入布尔查询的内容:')
            query = query.lower()
            # 识别替换括号，去除两侧空白字符
            query = re.sub('\(', ' ( ', query)
            query = re.sub('\)', ' ) ', query).strip()
            # 切词
            keywords = re.split(r'\s+', query)
            if keywords:
                bool_search.search_index(keywords, doc_map, n)
            else:
                print("Input does not contain keywords!")

        elif choose == '2':
            query = input('请输入语义查询的内容:')
            tokens_stem = semantic.query_process(query)     # 预处理查询query
            doc, relevancy_top10 = semantic.semantic_search(tokens_stem, n, terms, doc_count)   # 得到相关度top10的文档编号及其相关度
            print('\nThe top 10 most relevant documents:')
            for i in range(len(doc)):
                print(i+1, ':', doc_map[str(doc[i])], "\t\t relevancy: ", relevancy_top10[i])
            print('\n')

        else:
            print("\nInput ERROR!!!\n")

        print("The retrieval method is as follows:")
        print("0) quit   1) bool search   2) semantic search")
        choose = input("Please enter the retrieval method again:")


if __name__ == '__main__':
    main()