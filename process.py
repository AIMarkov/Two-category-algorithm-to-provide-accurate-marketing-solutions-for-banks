import numpy as np
from collections import deque
import random
class DATA():
    def __init__(self,data):
        self.data=data
        self.datalen = len(self.data["ID"])
        self.traindata = deque(maxlen=self.datalen + 1)
        print("data length：", self.datalen)
        print('columns:\n', self.data.columns)
        print("columns len:\n", len(self.data.columns))
        print("Head 2:\n", self.data.head(2))
    def samplpe(self,batch_num):
        batch=random.sample(self.traindata,batch_num)
        x,y=zip(*batch)
        return x,y

    def processed(self):
        propertydict={}
        #这里我们将各个非数值属性转换为数值
        #数值属性转为list，并放入属性字典
        IDvalue=list(self.data["ID"])
        propertydict["ID"]=IDvalue
        agevalue=list(self.data["age"])
        propertydict["age"]=agevalue
        balancevalue=list(self.data["balance"])
        propertydict["balance"]=balancevalue
        dayvalue=list(self.data["day"])
        propertydict["day"]=dayvalue
        durationvalue=list(self.data["duration"])
        propertydict["duration"]=durationvalue
        campaignvalue=list(self.data["campaign"])
        propertydict["campaign"]=campaignvalue
        pdaysvalue=list(self.data["pdays"])
        propertydict["pdays"]=pdaysvalue
        previousvalue=list(self.data["previous"])
        propertydict["previous"]=previousvalue
        yvalue=list(self.data["y"])
        propertydict["y"]=yvalue
        #非数值转换为数值,并构建成属性字典
        for column in self.data.columns:
            if column not in ["ID","age","balance","day","duration","campaign","pdays","previous","y"]:
                List=list(self.data[column])
                Set=set(List)
                Dict={}
                value=0
                for key in Set:
                    Dict[key]=value
                    value+=1
                Value=[]
                for term in List:
                    Value.append(Dict[term])
                propertydict[column]=Value
        #归一化数据集：
        for Property in self.data.columns:
            if Property not in ["ID","y"]:
                Propertyarray=np.asarray(propertydict[Property])
                Mean=sum(Propertyarray)/len(Propertyarray)
                Std=np.std(Propertyarray)
                propertydict[Property]=(Propertyarray-Mean)/Std

        #构建训练集deque，类似于经验池
        propertylist=[]
        for i in range(25317):
            for Property in self.data.columns:
                if Property not in ["ID","y"]:
                    propertylist.append(propertydict[Property][i])
            if propertydict['y'][i]==0:#属于第一类，不买
                self.traindata.append((np.asarray(propertylist),np.asarray([1,0])))
            else:#第二类买
                self.traindata.append((np.asarray(propertylist), np.asarray([0, 1])))
            propertylist = []







