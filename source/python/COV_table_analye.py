import csv
import pandas as pd
import numpy as np
import math

def preprocessing(table, label):
    res_dict = {}
    for con in label:
        if con == "Id" or con == "groupId" or con == "matchId" or con == "matchType" or con == "maxPlace" or con == "winPlacePerc":
                continue
        if con == "rideDistance" or con == "swimDistance" or con == "walkDistance":
                continue
        else :
            res_dict[con] = []
    res_dict["Distance"] = []
    for con in table:
        Distance = 0
        for index in label:
            if index == "Id" or index == "groupId" or index == "matchId" or index == "matchType" or index == "maxPlace" or index == "winPlacePerc":
                continue
            if index == "rideDistance" or index == "swimDistance" or index == "walkDistance":
                Distance += float(con[label.index(index)])
            else :
                try:
                    res_dict[index].append(int(con[label.index(index)]))
                except:
                    res_dict[index].append(float(con[label.index(index)]))
        res_dict["Distance"].append(Distance)
    new_labels = ["assists","boosts","damageDealt","DBNOs","headshotKills","heals","killPlace","killPoints","kills","killStreaks","longestKill","matchDuration","numGroups","rankPoints","revives","Distance","roadKills","teamKills","vehicleDestroys","weaponsAcquired","winPoints"]
    return res_dict, new_labels


def avg(Lis):
    Sum = 0
    for con in Lis:
        Sum += con
    return Sum/len(Lis)

def cov_XY(table, X1, X2, mean_table):

    Muw_1 = mean_table[X1]
    Muw_2 = mean_table[X2]

    COV_12 = 0

    for i in range(len(table[X1])):
        COV_12 += (table[X1][i] - Muw_1)*(table[X2][i] - Muw_2)
    
    return COV_12/len(table[X1])

def Deviation(RVari, mean):
    Sum = 0
    average = mean
    for con in RVari:
        Sum += (con - average)*(con - average)
    return  math.sqrt(Sum / len(RVari))

F = open('./train_solo_V2.csv','r',encoding='utf-8')
rdr = csv.reader(F)
data_table = []
for line in rdr:
    data_table.append(line)

labels = data_table[0]
del data_table[0]

new_data_Dic, new_label = preprocessing(data_table, labels)

length = len(new_data_Dic[new_label[1]])

new_data_table = []
#new_data_table.append(new_label)

for i in range(length):
    tmp = []
    for index in new_data_Dic:
        tmp.append(new_data_Dic[index][i])
    new_data_table.append(tmp)

res = pd.DataFrame(new_data_table, columns=new_label)
res.to_csv("./train_solo_V3.csv", header=True, index=False)
print("Complete")

'''
Pearson_table = []

mean_table = {}
for index in new_data_Dic:
    mean_table[index] = avg(new_data_Dic[index])

for i in new_label:
    tmp = []
    for j in new_label:
        cov_ij = cov_XY(new_data_Dic,i,j,mean_table)
        dev_i = Deviation(new_data_Dic[i], mean_table[i])
        dev_j = Deviation(new_data_Dic[j], mean_table[j])
        if not dev_i == 0 and not dev_j == 0 :
            tmp.append(cov_ij/(dev_i*dev_j))
        else :
            tmp.append(0)
    Pearson_table.append(tmp)

'''


'''
print(Pearson_table)
result = pd.DataFrame(Pearson_table,columns = new_label,index = new_label)

result.to_csv("./solo_Pearson.csv", header=True, index=True)

'''
