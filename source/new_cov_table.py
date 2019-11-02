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
                res_dict[index].append(con[label.index(index)])
        res_dict["Distance"].append(Distance)
    new_labels = ["assists","boosts","damageDealt","DBNOs","headshotKills","heals","killPlace","killPoints","kills","killStreaks","longestKill","matchDuration","numGroups","rankPoints","revives","Distance","roadKills","teamKills","vehicleDestroys","weaponsAcquired","winPoints"]
    return res_dict, new_labels



F = open('./train_duo_fpp_V2.csv','r',encoding='utf-8')
rdr = csv.reader(F)
data_table = []
for line in rdr:
    data_table.append(line)

labels = data_table[0]
del data_table[0]

new_data_Dic, new_label = preprocessing(data_table, labels)

length = len(new_data_Dic[new_label[1]])

new_data_table = []

index_labels = []
flag = True

for i in range(length):
    tmp = []
    for index in new_data_Dic:
        if flag :
            index_labels.append(index)
        tmp.append(new_data_Dic[index][i])
    flag = False
    new_data_table.append(tmp)

res = pd.DataFrame(new_data_table, columns=index_labels)
res.to_csv("./train_duo_fpp_V3.csv", header=True, index=False)
print("Complete")

#print(Pearson_table)
#result = pd.DataFrame(Pearson_table,columns = new_label,index = new_label)

#result.to_csv("./solo_Pearson.csv", header=True, index=True)


