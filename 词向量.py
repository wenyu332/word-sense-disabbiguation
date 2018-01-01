import jieba
import os
from gensim.models import word2vec
from gensim import models
import re
import numpy as np
import math
from nltk import wordnet
import nltk
FILEPATH='f:/NLP_project/TextInfoExp-master/Part1_TF-IDF/data/title_and_abs/'
CW=['NN','VBD','ADJ','ADV','VBG','NNS','VB','JJ','VBN']
#NN NNS 名词及其复数
#VB VBD(ed) VBG(ing) VBN(动词过去分词)
#ADJ ADV JJ(形容词)
#生成词向量

def generateWordVector(filePath):
    #word2vec模块使用
    if os.path.exists('wordsVector.model'):
        wordsVector=models.Word2Vec.load('wordsVector.model')
    else:
        print('第一次训练')
        sentences = word2vec.Text8Corpus(u"text8/text8")  # 加载语料
        for sentence in sentences:
            print(sentence)
        wordsVector=models.Word2Vec(sentences,size=200)
        wordsVector.save('wordsVector.model')
    return wordsVector

    #统计每个单词所对应的所有语义及其对应的注释中的单词
def countSenses(word):
    synsets=wordnet.wordnet.synsets(word)#每个word的近义词集合
    wordSynstets={}
    for synset in synsets:
        define=synset.definition()#单词定义
        example=synset.examples()#单词例子
        definitions = []
        examples = []
        for w in re.findall('[\w]*',define):
            if w is not '':
                definitions.append(w)
        for ex in example:
            for w in re.findall('[\w]*', ex):
                if w is not '':
                    examples.append(w)
        wordSynstets.update({synset:(definitions,examples)})
    return (word,wordSynstets)
#统计所有实词，生成候选集
def getCW(wordSeynstets):
    senseCW={}
    for synste in wordSeynstets[1]:
        defiCW = nltk.pos_tag(wordSeynstets[1][synste][0])
        examCW=nltk.pos_tag(wordSeynstets[1][synste][1])
        CWsets=[]
        for word,tag in defiCW:
            if tag in CW and word!=wordSeynstets[0]:
                CWsets.append(word)
        for word, tag in examCW:
            if tag in CW and word!=wordSeynstets[0]:
                CWsets.append(word)
        senseCW.update({synste:CWsets})
    return (wordSeynstets[0],senseCW)
#计算大于deta的余弦相似度
def getDeatCandi(senseCW,wordsVector=None,threshold=None):
    senseCandWords={}
    candWords=[]
    #计算余弦相似度
    centerVector = wordsVector[senseCW[0]]
    centerVector=np.array(centerVector)
    centerSum=math.sqrt(np.matmul(centerVector.T,centerVector))
    for sense in senseCW[1]:
        for word in senseCW[1][sense]:
            wordVector=wordsVector[word]
            wordVector = np.array(wordVector)
            wordSum = math.sqrt(np.matmul(wordVector.T, wordVector))
            similarty=np.matmul(centerVector.T,wordsVector)/(centerSum*wordSum)
            if similarty>threshold:
                candWords.append(word)
        senseCandWords.update({sense:candWords})
    return  (senseCW[0],senseCandWords)
#生成语义向量
def generateSenseVector(wordsVector,senseCandWords):
    senseVector={}
    for sense in senseCandWords[1]:
        vecSum = 0
        count=0
        for word in sense:
            count+=1
            vecSum+=wordsVector[word]
        vec=vecSum/count#计算每个语义的向量 公式没有看懂，在理解当中
        senseVector.update({sense:vec})
    return senseVector

#采用S2C算法进行词义消歧
def disambiguation(sentence,wordsVector=None,senseVector=None,ksi=None):
    seperate=[]
    words=re.findall('[a-zA-Z]*',sentence)
    for word in words:
        if word is not '':
            seperate.append(word)
    pos=nltk.pos_tag(seperate)
    POSwords=[]
    #tag[0]为单词 tag[1]为词性
    for tag in pos:
        if tag[1]  in CW:
            POSwords.append(tag[0])
    CWsum=0
    for word in POSwords:
        CWsum+=wordsVector[word]
    contextVec=CWsum/len(POSwords)
    senseCount={}
    for word in POSwords:
        synsets = wordnet.wordnet.synsets(word)
        if len(synsets)>1:
            senseCount.update({len(synsets):word})
    count=sorted(senseCount)
    for i in count:
        data=countSenses(senseCount.get(i))
        CWdata=getCW(data)
        getDeatCandi(CWdata,wordsVector,ksi)

        #print(CWdata)
        for synset in CWdata[1]:
            print(synset,CWdata[1].get(synset))
disambiguation('He sat on the bank of the lake.')


'''
wordsVector=generateWordVector(FILEPATH)
print(wordsVector['positive'])

#wordsVector=generateWordVector(FILEPATH)
senseSets=countSenses('bank')
CWsets=getCW(senseSets)
#generateSenseVector(wordsVector,CWsets)
'''