import csv
import pandas as pd
import numpy as np
import math

'''
            elif index == "winPlacePerc":
                try:
                    win_pre.append(float(table[i][label.index(index)]))
                except:
                    win_pre.append(0)
'''

def preprocessing(table, label):
    new_labels = ["assists","boosts","damageDealt","DBNOs","headshotKills","heals","killPlace","killPoints","kills","killStreaks","longestKill","matchDuration","numGroups", "rankPoints","revives","Distance","roadKills","teamKills","vehicleDestroys","weaponsAcquired","winPoints","winPlacePerc"]

    DATA_TABLE = np.zeros((len(table),len(new_labels)))

    #win_pre = []

    for i in range(len(table)):
        Distance = 0
        for index in label:
            if index == "Id" or index == "groupId" or index == "matchId" or index == "matchType" or index == "maxPlace":
                continue
            elif index == "rideDistance" or index == "swimDistance" or index == "walkDistance":
                Distance += float(table[i][label.index(index)])
            else :
                try :
                    DATA_TABLE[i][new_labels.index(index)] = int(table[i][label.index(index)])
                except :
                    DATA_TABLE[i][new_labels.index(index)] = float(table[i][label.index(index)])
        if DATA_TABLE[i][new_labels.index("Distance")] == 0:
            DATA_TABLE[i][new_labels.index("Distance")] = Distance
    res_dict = pd.DataFrame(DATA_TABLE, columns=new_labels)
    return DATA_TABLE, res_dict, new_labels

F = open('./train_solo_V2.csv','r',encoding='utf-8')
rdr = csv.reader(F)
data_table = []
for line in rdr:
    data_table.append(line)

labels = data_table[0]
del data_table[0]

new_TABLE ,new_data_Dic, new_label = preprocessing(data_table, labels)

K = np.array(new_data_Dic["Distance"])
#print(K)
#print(W)

res = pd.DataFrame(new_TABLE, columns=new_label)
res.to_csv("./V5_data/train_solo_V5.csv", header=True, index=False)
print("Complete")

#print(Pearson_table)
#result = pd.DataFrame(Pearson_table,columns = new_label,index = new_label)

#result.to_csv("./solo_Pearson.csv", header=True, index=True)


