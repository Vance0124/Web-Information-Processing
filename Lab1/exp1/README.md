## lab1-邮件搜索引擎

<center>曾勇程  &emsp; PB18000268</center>

<center>张弛 &emsp; PB18000269</center>

### 实验运行环境

1. 硬件条件：一台PC机(联想拯救者Y7000)

2. 软件条件：

   系统Windows10

   语言python3.8.5(64位)

   编辑器pycharm2018.3.4



### 编辑运行方式

可在编辑器pycharm中或其他支持python的编辑器中运行(保证库已经`pip`下来)。



### 文件含义

```
exp1/
|----src/
	|----pretreatment.py 		//预处理源代码, 用于生成倒排表和tf_idf文档等文件
	|----bool_search.py			//布尔检索源代码
	|----semantic_search.py		//语义检索源代码
	|----mail_retrieval_system.py 
								//将布尔检索和语义检索整合到一个文件
|----dataset/
	|----enron_mail_20150507	//数据集文件夹, 这里不上传该数据集
|----output/
	|----doc_count.txt			//存储top1000词项的文档频率
	|----doc_map.txt			//以字典形式存储每封邮件(共517401封)的路径,方便查询
	|----inverse_index.txt		//倒排索引表,形式: 词项 文档频率 文档索引
	|----tf_idf.txt				//存储tf_idf矩阵,每个文档的tf_idf占一行,奇数项为词项索引,偶数项为词项的tf_idf值
	|----top1kwords.txt			//存储top1000的词项
|----实验报告.pdf
|----README
```



### 关键函数说明

#### 预处理部分

##### 获得所有文档路径

代码如下：

~~~python
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
~~~

​	采用深度遍历算法，先查询当前目录下的所有文件，然后对文件进行判断，若为目录，则对该目录采用深度遍历，若为一封邮件，则将它的路径添加到列表中，本函数采用相对路径，从`path`开始深度遍历。

将结果(`file_list`)保存到`filelist_path`所指代的文件中(在`pretreatment()`函数中执行该操作)：

~~~python
	# 打印所有文档路径
    document_map = dict(zip(range(len(filelist)), filelist)) # 转化为字典
    f = open(filelist_path, 'w')
    handler = json.dumps(document_map)			# 调用 json 库
    f.write(handler)
    f.close()
~~~



`pretreatment.py`中的`pretreatment(filelist, target_path, filelist_path)`主要的任务是**提取邮件有用内容**、**分词**和**词根化**3部分，因此下面分3部分讲解该函数。

##### 提取邮件有用内容

使用`re`库，通过正则表达式获取每封邮件中的标题和内容正文，由于有些邮件中不是以`utf-8`格式编码（即可能有中文），故在打开方式中，增加参数`errors='ignore'`，忽略打开错误。

~~~python
    find_content = "\n\n(.*)"			# 匹配内容正文
    find_subject = "subject: (.*?)\n"	# 匹配标题
    with open(target_path, 'w') as wfp:
    	for file in filelist:
            fp = open(file, 'r', errors='ignore')
            txt = fp.read().lower()     # 将文本全转化为小写
            fp.close()
            subject = re.findall(find_subject, txt, re.M | re.I)[0]      # 得到主题
            content = re.findall(find_content, txt, re.S)[0]      # 得到邮件正文内容
~~~



##### 分词

分词前我们会对读取的正文内容做初步处理，即把**标点符号**和**数字**去除，为防止影响到之后的分词(数字的存在会干扰`top1000`词项的选取)，我们将其替换为空格。

~~~python
    string_replace = string.punctuation + '0123456789'
    replacement = ''  # 用 ' ' 替换 string_repalce 中的每个符号
    for i in range(len(string_replace)):
        replacement = replacement + ' '	
~~~

~~~python
    		remove = str.maketrans(string_replace, replacement)     # 构造标点符号和数字到空格的映射关系 
    		content = content.translate(remove)						# 去除标点符号和数字
            subject = subject.translate(remove)
~~~

原本我们使用nltk库中的nltk.word_tokenize()函数进行分词，但考虑到文档数目51万过大，且内容均为英文，为了提升预处理的效率，我们改为使用`re`库中的分割函数`split()`直接对空白字符进行切分。

~~~python
            # 由于使用nltk库分词会大大降低对51w封邮件的处理效率，改为使用split分词
            # tokens = nltk.word_tokenize(content)        # 分词 spilt
            # sub_tokens = nltk.word_tokenize(subject)
            tokens = content.split()
            sub_tokens = subject.split()
~~~



##### 去除停用词

第一次尝试中原本想利用`nltk`库中的停用词表，如下：

~~~python
import nltk
from nltk.corpus import stopwords
~~~

~~~python
        	# 去除停用词
    		tokens_nosw = [word for word in tokens if word not in stopwords.words('english')]
    		sub_tokens_nosw = [word for word in sub_tokens if word not in stopwords.words('english')]           
~~~

但是经过试验发现，采用停用词库会大大降低对51w封邮件的预处理效率（导致可能会运行一整天）。并且在语义查询时，如果只输入一句话，去除停用词后会有很大的误导性，导致语义查询出来的邮件中只有几个词和查询`query`相同，但语义完全不同，在权衡以后，**我们决定不采用停用词表**，这样的话，我们能将预处理的速度能缩减为2小时。



##### 词根化

调用`nltk.stem`库中的`SnowballStemmer('english')`类，参数为选择的语言

~~~python
import nltk.stem
~~~

~~~python
    stem = nltk.stem.SnowballStemmer('english')  # 参数是选择的语言, 用于词根化
~~~

~~~python
			tokens_stem = [stem.stem(ws) for ws in tokens]     # 词根化
    		sub_tokens_stem = [stem.stem(ws) for ws in sub_tokens]
~~~



##### 生成邮件预处理文件

将预处理完的邮件内容保存到`target_path`所指的文件中：

~~~python
 			content_fin = ' '.join(tokens_stem)           # 用空格拼接内容
            subject_fin = ' '.join(sub_tokens_stem)
            mail = subject_fin + ' ' + content_fin + '\n\n' + '-----Document demarcation-----' + '\n\n'   														# 以 '-----Document demarcation-----' 划分每个文档
            wfp.write(mail)             # 写入预处理文档
~~~



##### 统计top1000词项

利用邮件预处理文件，记录出现的词项和频率，取出`top1000`的词项，输出到 `txt_2`文件中。

~~~python
def get_top_1k_words(txt_1, txt_2):
    global terms
    wordlist = {}		# 记录出现过的词项和频率
    fd = open(txt_1, 'r')
    line = fd.readline()
    while line:
        if line != "-----Document demarcation-----\n":	# 判断一封邮件是否结束
            words = line.split()	# 分词
            for word in words:      
                if word in wordlist:
                    wordlist[word] += 1		# 频率加1
                else:
                    wordlist[word] = 1		# 记录出现的新词
        line = fd.readline()
    fd.close()
    f2 = open(txt_2, 'w')			# 写入 txt_2 文档
    for i in range(1000):
        word_top = max(wordlist, key=wordlist.get)      # 用 max 取出字典中的 top1000 的键
        terms.append(word_top)
        f2.write(word_top + '\n')
        wordlist[word_top] = -1	 		# 将已经取出的词项的频率置为 -1
    f2.close()
~~~



##### 建立倒排表文件和词项、文档频率文件

扫描邮件预处理文件，用列表 `count`记录每篇文档中的词项频率，用列表`flag`标记每篇文档中出现的top1000词项是否已经添加到倒排表中。将词项频率写入`out_doc_count_txt`文件中，由于词项频率为**稀疏矩阵**，故只记录每篇文档出现过的`top1000`的词项的频率，存储形式为：**“词项	对应的词项频率	”**。将倒排表写入`path1`指定的文件中。 用`doc_count`统计文档频率，并写入`path2`指定的文件中。

~~~python
def create_inve_table(txt_1, out_doc_count_txt):
    global terms,  inve_table   # terms 为 top1000 词项, inve_table 为 倒排表(字典)
    fd = open(txt_1, 'r')
    line = fd.readline()
    doc_id = 0                  # 文档编号
    count = np.zeros(1000)      # 统计 top1000 的词项在每篇文档中的出现次数
    flag = np.zeros(1000)         # 用于标记1000个词是否已经在文档中出现过
    wfd = open(out_doc_count_txt, 'w')
    while line:
        if line != "-----Document demarcation-----\n":		# 判断一封邮件是否结束
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
~~~



##### 建立tf_idf矩阵文件

利用所学公式，计算每篇文档中出现过的`top1000`词项的tf_idf值,  注意：这里词项频率我们没有建立二维全矩阵，而是采用了压缩矩阵的存储方法，以降低存储和搜索开销。在该文件中，存储形式为每一行对应一个文档(一封邮件)，并且按 **"词项 	词项频率 	"**形式存储，为了提高速率，我们计算tf_idf值时要计算词项的下标以及该词项对应的词项频率的下标。`tf_idf`矩阵文件也采用压缩矩阵存储形式(`tf_idf`矩阵也是个稀疏矩阵)，即每一行对应一个文档(一封邮件)，并且按 **"词项 	对应的tf_idf值 	"**形式存储。

~~~python
def doc_tf_idf(n, tc_path, output_txt):  # n 表示文档总数
    global doc_count
    fd = open(tc_path, 'r')			# tc_path 为记录词项频率的文件的路径
    line = fd.readline()
    wfd = open(output_txt, 'w')
    while line:
        doc_term = line.split()
        nums = int(len(doc_term)/2)	# 计算词项数量(存储形式："词项 词项频率 ",即从下标0开始,偶数项为词项,奇数项为词项频率)
        element = ''
        for term_id in range(nums):           # 只记录了词项频率非0的词项, 因此 tf 不可能为 0
            if doc_count[int(doc_term[2*term_id]) - 1] != n:  # 等于 n 的话, tf_idf 的值就为 0, 不需要存储
                tf_idf = round((1 + math.log(float(doc_term[2*term_id+1]), 10)) * math.log(n/doc_count[int(doc_term[2*term_id]) - 1], 10), 6)		# 计算该文档中出现过的top1000词项的tf_idf值
                element = element + str(doc_term[2*term_id]) + ' ' + str(tf_idf) + ' '
        element = element + '\n'
        wfd.write(element)
        line = fd.readline()
    wfd.close()
    fd.close()
~~~



##### 清除无用文件

上面一共建立了7个文件，当所有文件建立完以后，**邮件预处理文件**和**词项频率文件**就没用了，可以手动删除以节省空间。



#### 布尔检索

布尔搜索需要先读取倒排索引表。

~~~python
# 从文件中读取索引创建索引字典
def create_index(file_name):
    global inverse_index
    f = open(file_name, 'r')
    line = f.readline().rstrip()
    while line:
        line = re.split(r'\s+', line)
        inverse_index[line[0]] = set()	# 用集合表示含有该词项的文档集合
        for id in line[2:]:		# 0 为 词项, 1 为文档频率, 2 之后为文档编号
            inverse_index[line[0]].add(id)
        line = f.readline().rstrip()
    return inverse_index
~~~



随后对于每次输入的搜索语句，计算其布尔逻辑运算结果，得到返回文档序列(在`search_index(keywords, doc_map, n)`函数中)。我们使用了两个栈分别存放**操作码**（and, not, or, 左右括号）和**操作数**（即查询单词），通过栈来实现布尔表达式的优先级计算(默认优先级：not > and > or)。

~~~python
	global inverse_index
    opcode = []		# 用opcode列表充当操作码栈
    operand = [] 	# 用operand列表充当操作数栈
    full = set()
    for id in range(n):		#构建全集
        full.add(str(id))
    for token in keywords:
        # 如果读取左括号或not，直接入栈
        if token == '(' or token == 'not':
            opcode.append(token)
        # 如果读取运算符and，判断栈顶运算符是否为and，若是则计算，若不是则直接入栈
        elif token == 'and':
            if len(opcode) and opcode[len(opcode) - 1] == 'and':
                preval = operand.pop()
                pre2val = operand.pop()
                operand.append(preval & pre2val)
            else:
                opcode.append(token)
        # 如果读取or，因为其优先级较低，在栈顶为and或or时都先计算栈顶运算再入栈
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
        # 若读取到右括号，则不断循环计算栈顶运算，直到遇到匹配的左括号为止
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
            # 计算完括号后，若栈顶为not，直接计算
            if len(opcode) and opcode[len(opcode) - 1] == 'not':
                opcode.pop()
                preval = operand.pop()
                operand.append(full - preval)
        # 若读取的为单词，仅在栈顶为not时直接计算，否则直接入栈
        else:
            if len(opcode) and opcode[len(opcode) - 1] == 'not':
                opcode.pop()
                operand.append(full - inverse_index.get(token, set()))
            else:
                operand.append(inverse_index.get(token, set()))
    # 计算最后可能剩下的二元表达式
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
~~~

需要注意的是，对于输入的查询语句，我们同样会先对其进行词根化处理，来和倒排表中单词匹配。

~~~python
    stem = nltk.stem.SnowballStemmer('english')
    keywords = [stem.stem(ws) for ws in keywords]
~~~



布尔检索主函数：

~~~python
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
~~~



#### 语义检索

##### 建立tf_idf矩阵

以**二维矩阵**形式存储，每个子列表表示一个文档(一封邮件)，子列表内部，以下标0开始，下标为`2k`的列表元素为词项编号，而下标为`2k+1`的列表元素为该词项对应的`tf_idf`值，大大缩小矩阵存储开销，也减少了无用计算。结果存储在全局列表`dtifs`中。

~~~python
def get_tf_idf(path):
    global dtifs
    fd = open(path, 'r')
    line = fd.readline()
    while line:
        if line != '\n':
            doc_tf = line.split()
            dtifs.append(doc_tf)
        else:		# 为 '\n' 表示该邮件中, top1000词项都没有出现
            doc_tf = []		
            dtifs.append(doc_tf)
        line = fd.readline()
    fd.close()
~~~



##### 预处理查询query

我们所支持的语义查询是任意输入一行语句，可以包括**任意的标点符号**，搜索相关度前10的文档(邮件)，因此要对查询`query`做预处理：转化为**全小写**、**替换标点符号和数字**、**词根化**和**分词**。

~~~python
def query_process(query):
    # 对查询做预处理
    query = query.lower()	# 转化为小写
    string_replace = string.punctuation + '0123456789'
    replacement = ''  # 用 ' ' 替换 string_replace 中的每个符号
    for i in range(len(string_replace)):
        replacement = replacement + ' '
    remove = str.maketrans(string_replace, replacement)  # 去除标点符号和数字
    query = query.translate(remove)
    words = query.split()	# 分词
    stem = nltk.stem.SnowballStemmer('english')  # 参数是选择的语言, 用于词根化
    tokens_stem = [stem.stem(ws) for ws in words]  # 词根化
    return tokens_stem
~~~



##### 语义查询实现

语义查询主要在`semantic_search(query, n, terms, doc_counts)`函数中实现，下面是他的每一个部分。

根据老师课上讲的原理，对查询语句先计算出查询query中top1000词项的tf-idf值，形成tf-idf向量：

~~~python
 	vec_q = np.zeros(1000)		# 初始化
    for term in query:
        if term in terms:
            vec_q[terms.index(term)] += 1.0	 # 统计每个词在查询query中的出现次数
    for i in range(len(vec_q)):
        if vec_q[i] != 0:
            vec_q[i] = round((1 + math.log(vec_q[i], 10)) * math.log(n/float(doc_counts[i]), 10), 10)
            								 # 计算查询query中top1000词项的 tf_idf 值
	vec_q_np = np.array(vec_q)			# 转化为 numpy库的数组               
~~~



得到每篇文档对应的tf-idf向量(`dtifs`采用压缩矩阵的存储方法：以**二维矩阵**形式存储，每个子列表表示一个文档(一封邮件)，子列表内部，以下标0开始，下标为`2k`的列表元素为词项编号，而下标为`2k+1`的列表元素为该词项对应的`tf_idf`值)：

~~~python
global dtifs
~~~

~~~python
    for tds in dtifs:
        vec_doc = np.zeros(1000)
        nums = int(len(tds)/2)
        for term_id in range(nums):
            vec_doc[int(tds[2 * term_id]) - 1] = float(tds[2 * term_id + 1]) 
        vec_doc_np = np.array(vec_doc)
~~~



用查询query的**tf-idf向量**与各个文档**tf-idf向量**的做内积运算，并做**归一化**，得到相关度列表`relevancy`。

~~~python
        value = np.sqrt((vec_q_np * vec_q_np).sum() * (vec_doc_np * vec_doc_np).sum())	# 归一化系数
        if value != 0:		
            cosine = ((vec_q_np * vec_doc_np).sum())/value  # 求相关度
        else:		# 分母为 0 表示查询query或该文档中午top1000词项出现, 相关度自然为 0 
            cosine = 0
        relevancy.append(cosine)                       # 求得相关度的列表
~~~



得到相关度top10的文档的编号及其相关度：

~~~python
    for i in range(10):
        index = relevancy.index(max(relevancy))
        doc.append(index)
        relevancy_top10.append(relevancy[index])
        relevancy[index] = -1       # -1 表示已经用完,不对后面的分析产生影响
    return doc, relevancy_top10
~~~



语义查询主函数：

~~~python
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
        tokens_stem = query_process(query)
        doc, relevancy_top10 = semantic_search(tokens_stem, n, terms, doc_count)
        print('The top 10 most relevant documents:')
        for i in range(len(doc)):
            print(i+1, ':', doc_map[str(doc[i])], "\t\t relevancy: ", relevancy_top10[i])
~~~



#### 整合的邮件检索系统

##### 导入布尔检索和语义检索源代码

~~~python
import semantic_search as semantic
import bool_search
~~~



##### 邮件检索系统

邮件检索系统采用了多线程优化加载`loading`速率。

~~~python
import re
import threading

def main():
    filelist_path = '../output/doc_map.txt'
    doc_count_txt = '../output/doc_count.txt'
    tf_idf_txt = '../output/tf_idf.txt'
    inve_txt = '../output/inverse_index.txt'   
    terms_txt = '../output/top1kwords.txt'

    print("Please wait a few minutes for loading...")

    t1 = threading.Thread(target=semantic.get_tf_idf, args=(tf_idf_txt,))  # 创建线程 1
    t1.start()  # 开始执行线程 1
    t2 = threading.Thread(target=bool_search.create_index, args=(inve_txt,))    # 创建线程 2
    t2.start()      # 开始执行线程 2

    doc_map = semantic.create_map(filelist_path)  # 创建映射
    doc_count = semantic.get_doc_count(doc_count_txt)  # 创建文档频率 doc_counts
    terms = semantic.get_terms(terms_txt)  # 创建 top1000 的词项

    t1.join()       # 等待线程 1 和 2 执行完
    t2.join()

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
~~~

