import numpy as np
import csv
import pandas as pd
import math
import zipfile

def data_prepro(data, label):
    res = []
    for con in data:
        tmp = []
        tmp.append(int(con[label.index("assists")]))
        tmp.append(int(con[label.index("boosts")]))
        tmp.append(float(con[label.index("damageDealt")]))
        tmp.append(int(con[label.index("DBNOs")]))
        tmp.append(int(con[label.index("headshotKills")]))
        tmp.append(int(con[label.index("heals")]))
        tmp.append(int(con[label.index("killPlace")]))
        tmp.append(int(con[label.index("killPoints")]))
        tmp.append(int(con[label.index("kills")]))
        tmp.append(int(con[label.index("killStreaks")]))
        tmp.append(float(con[label.index("longestKill")]))
        tmp.append(int(con[label.index("matchDuration")]))
        tmp.append(int(con[label.index("numGroups")]))
        tmp.append(int(con[label.index("rankPoints")]))
        tmp.append(int(con[label.index("revives")]))
        tmp.append(float(con[label.index("rideDistance")]) + float(con[label.index("swimDistance")]) + float(con[label.index("walkDistance")]))
        tmp.append(int(con[label.index("roadKills")]))
        tmp.append(int(con[label.index("teamKills")]))
        tmp.append(int(con[label.index("vehicleDestroys")]))
        tmp.append(int(con[label.index("weaponsAcquired")]))
        tmp.append(int(con[label.index("winPoints")]))
        res.append(tmp)
    new_labels = ["assists","boosts","damageDealt","DBNOs","headshotKills","heals","killPlace","killPoints","kills","killStreaks","longestKill","matchDuration","numGroups","rankPoints","revives","Distance","roadKills","teamKills","vehicleDestroys","weaponsAcquired","winPoints"]
    #new_labels = ["assists","boosts","damageDealt","DBNOs","headshotKills","heals","killPlace","killPoints","kills","killStreaks","longestKill","matchDuration","numGroups","rankPoints","revives","Distance","roadKills","vehicleDestroys","weaponsAcquired","winPoints"]
    return res, new_labels

def avg(Lis):
    Sum = 0
    for con in Lis:
        Sum += con
    return Sum/len(Lis)

def cov_XY(table, X1, X2):

    data = np.transpose(table)
    data_X1 = data[X1]
    data_X2 = data[X2]

    Muw_1 = avg(data_X1)
    Muw_2 = avg(data_X2)

    COV_12 = 0

    for i in range(len(data_X1)):
        COV_12 += (data_X1[i] - Muw_1)*(data_X2[i] - Muw_2)
    
    return COV_12/len(data_X1)

def Deviation(RVari):
    Sum = 0
    average = avg(RVari)
    for con in RVari:
        Sum += (con - average)*(con - average)
    return  math.sqrt(Sum / len(RVari))

F = open('./train_solo_fpp_V2.csv','r',encoding='utf-8')
rdr = csv.reader(F)
data_table = []
for line in rdr:
    data_table.append(line)

labels = data_table[0]
del data_table[0]

new_data_table, new_label = data_prepro(data_table, labels)

Pi_table = []

for i in range(len(new_label)):
    tmp  = []
    for j in range(len(new_label)):
        T_table = np.transpose(new_data_table)
        cov_ij = cov_XY(new_data_table,i,j)
        dev_i = Deviation(T_table[i])
        dev_j = Deviation(T_table[j])
        tmp.append(cov_ij/(dev_i*dev_j))
    Pi_table.append(tmp)

#print(Pi_table)
print(pd.DataFrame(Pi_table,columns = new_label,index = new_label))
