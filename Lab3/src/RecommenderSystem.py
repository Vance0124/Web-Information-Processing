import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

trainDF = pd.read_csv("data/training.dat", sep=",", header=None,
    names=["UserID", "MovieID", "Rating", "Timestamp", "Tag1"])
# 可以继续加上Tag2, ...
testDF = pd.read_csv("data/testing.dat", sep=",", header=None,
    names=["UserID", "MovieID", "Timestamp", "Rating"])

Num2UserID = dict(enumerate(list(trainDF["UserID"].unique())))
UserID2Num = {key: value for value, key in Num2UserID.items()}
UserNum = len(UserID2Num)       # 2173

Num2MovieID = dict(enumerate(list(trainDF["MovieID"].unique())))
MovieID2Num = {key: value for value, key in Num2MovieID.items()}
MovieNum = len(MovieID2Num)     # 58431

DataMatrix = np.zeros((UserNum, MovieNum))
for line in trainDF.itertuples():
    DataMatrix[UserID2Num[line[1]], MovieID2Num[line[2]]] = line[3]+1
    # 将评分为0项与矩阵未评分项区分
DataExist = (DataMatrix != 0)
MovieMeanScore = DataMatrix.sum(axis=0) / DataExist.sum(axis=0)
# print(MovieMeanScore)
UserBias = DataMatrix.sum(axis=1) / DataExist.sum(axis=1)

SocialRelation = {}
readrel = open("data/relation.txt", "r")
attention = readrel.readline()
while attention:
    attention = attention.strip("\n")
    userid = int(attention.split(":", 1)[0])
    SocialRelation[userid] = set(map(int, attention.split(":", 1)[1].split(",")))
    attention = readrel.readline()
readrel.close()
# print(SocialRelation)

def ColdStart(userid, movieid):
    if userid in SocialRelation:
        count = 0
        ratingsum = 0
        for leaderid in SocialRelation[userid]:
            if DataMatrix[UserID2Num[leaderid], MovieID2Num[movieid]] != 0:
                ratingsum += DataMatrix[UserID2Num[leaderid], MovieID2Num[movieid]]
                count += 1
        if count != 0:
            return ratingsum / count
        else:
            return MovieMeanScore[MovieID2Num[movieid]]
    else:
        return MovieMeanScore[MovieID2Num[movieid]]

BiasMatrix = DataExist * UserBias[:, np.newaxis]
# print(BiasMatrix)
NormMatrix = DataMatrix - BiasMatrix
UserSimilarity = cosine_similarity(NormMatrix)
# print(UserSimilarity)

def UserCFPredict(userid, movieid):
    useri = UserID2Num[userid]
    movie = MovieID2Num[movieid]
    weight = 0
    rating = 0
    # k-nearest
    # K = 25
    # TopK = []
    # for index in range(K):
    #     TopK.append([0, 0])     # userid, similarity
    # for userj in range(UserNum):
    #     if DataMatrix[userj, movie] != 0:
    #         ptr = K-1
    #         if UserSimilarity[useri, userj] > TopK[ptr][1]:
    #             ptr = ptr-1
    #             while (ptr >= 0) and (UserSimilarity[useri, userj] > TopK[ptr][1]):
    #                 TopK[ptr+1] = TopK[ptr]
    #                 ptr = ptr-1
    #             TopK[ptr+1] = [userj, UserSimilarity[useri, userj]]
    # for index in range(K):
    #     if TopK[index][1] > 0:
    #         weight += TopK[index][1]
    #         rating += TopK[index][1]*NormMatrix[TopK[index][0], movie]
    # no k-nearest
    for userj in range(UserNum):
        if (DataMatrix[userj, movie] != 0) and (UserSimilarity[useri, userj] > 0):
            weight += UserSimilarity[useri, userj]
            rating += UserSimilarity[useri, userj]*NormMatrix[userj, movie]
    if weight == 0:
        return UserBias[useri]
    return UserBias[useri] + rating/weight

def DataSupplement(DataMatrix):
    nDataMatrix = np.zeros((UserNum, MovieNum))
    for user in range(UserNum):
        for movie in range(MovieNum):
            if DataMatrix[user, movie] != 0:
                nDataMatrix[user, movie] = DataMatrix[user, movie]
            else:
                userid = Num2UserID[user]
                movieid = Num2MovieID[movie]
                nDataMatrix[user, movie] = UserCFPredict(userid, movieid)
    return nDataMatrix
# nDataMatrix = DataSupplement(DataMatrix)
# np.save("SupplementedData", nDataMatrix)

def PrintResult(filename):
    writeres = open(filename, "w")
    for line in testDF.itertuples():
        userid = line[1]
        movieid = line[2]
        predscore = round(UserCFPredict(userid, movieid)-1)
        if predscore < 0:
            predscore = 0
        elif predscore > 5:
            predscore = 5
        writeres.write(str(predscore) + "\n")
    writeres.close()
filename = "submission21.txt"
# PrintResult(filename)

class SVD:
    def __init__(self,mat,K=30):
        self.mat=np.array(mat)
        self.K=K
        self.bi={}
        self.bu={}
        self.qi={}
        self.pu={}
        self.avg=np.mean(self.mat[:,2])
        for i in range(self.mat.shape[0]):
            uid=self.mat[i,0]
            iid=self.mat[i,1]
            self.bi.setdefault(iid,0)
            self.bu.setdefault(uid,0)
            self.qi.setdefault(iid,np.random.random((self.K,1))/10*np.sqrt(self.K))
            self.pu.setdefault(uid,np.random.random((self.K,1))/10*np.sqrt(self.K))

    def predict(self,uid,iid):  #预测评分的函数
        #setdefault的作用是当该用户或者物品未出现过时，新建它的bi,bu,qi,pu，并设置初始值为0
        self.bi.setdefault(iid,0)
        self.bu.setdefault(uid,0)
        self.qi.setdefault(iid,np.zeros((self.K,1)))
        self.pu.setdefault(uid,np.zeros((self.K,1)))
        rating=self.avg+self.bi[iid]+self.bu[uid]+np.sum(self.qi[iid]*self.pu[uid]) #预测评分公式
        #由于评分范围在1到5，所以当分数大于5或小于1时，返回5,1.
        if rating>6:
            rating=6
        if rating<1:
            rating=1
        return rating
    
    def train(self,steps=200,gamma=0.035,Lambda=0.15):    #训练函数，step为迭代次数。
        print('train data size',self.mat.shape)
        for step in range(steps):
            print('step',step+1,'is running')
            KK=np.random.permutation(self.mat.shape[0]) #随机梯度下降算法，kk为对矩阵进行随机洗牌
            rmse=0.0
            for i in range(self.mat.shape[0]):
                j=KK[i]
                uid=self.mat[j,0]
                iid=self.mat[j,1]
                rating=self.mat[j,2]
                eui=rating-self.predict(uid, iid)
                rmse+=eui**2
                self.bu[uid]+=gamma*(eui-Lambda*self.bu[uid])  
                self.bi[iid]+=gamma*(eui-Lambda*self.bi[iid])
                tmp=self.qi[iid]
                self.qi[iid]+=gamma*(eui*self.pu[uid]-Lambda*self.qi[iid])
                self.pu[uid]+=gamma*(eui*tmp-Lambda*self.pu[uid])
            gamma=0.95*gamma
            print('rmse is',np.sqrt(rmse/self.mat.shape[0]))
    
    def test(self,test_data):  #gamma以0.95的学习率递减
        print('test data size',test_data.shape)
        writeres = open(filename, "w")
        for i in range(test_data.shape[0]):
            uid=test_data[i,0]
            iid=test_data[i,1]
            predscore = round(self.predict(uid, iid)-1)
            writeres.write(str(predscore) + "\n")
        writeres.close()
            
def getData():   #获取训练集和测试集的函数
    train_data = []
    test_data = []
    for line in trainDF.itertuples():
        train_data.append([UserID2Num[line[1]], MovieID2Num[line[2]], line[3]+1])
    for line in testDF.itertuples():
        test_data.append([UserID2Num[line[1]], MovieID2Num[line[2]]])
    train_data=np.array(train_data)
    test_data=np.array(test_data)
    print('load data finished')
    print('train data ',len(train_data))
    print('test data ',len(test_data))
    return train_data,test_data
    
train_data,test_data=getData()
a=SVD(train_data,40)
a.train()
a.test(test_data)