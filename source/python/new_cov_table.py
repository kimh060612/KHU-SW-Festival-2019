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
"""
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
"""

def PREPROCESSING(table, labels, WH_D_L = "DICT"):

    res_dict = {}
    for con in labels:
        if con == "Id" or con == "groupId" or con == "matchId" or con == "matchType" or con == "maxPlace" or con == "rankPoints" or con == "killPoints" or con == "winPoints":
            continue
        else :
            res_dict[con] = []
    win_pre = []
    for con in table:
        for index in labels:
            if index == "Id" or index == "groupId" or index == "matchId" or index == "matchType" or index == "maxPlace" or index == "rankPoints" or index == "killPoints" or index == "winPoints":
                continue
            elif index == "winPlacePerc":
                try:
                    res_dict["winPlacePerc"].append(float(con[labels.index(index)]))
                except:
                    res_dict["winPlacePerc"].append(0)
            else :
                try:
                    res_dict[index].append(int(con[labels.index(index)]))
                except:
                    res_dict[index].append(float(con[labels.index(index)]))
    new_labels = ["assists","boosts","damageDealt","DBNOs","headshotKills","heals","killPlace","kills","killStreaks","longestKill","matchDuration","numGroups","revives","roadKills","teamKills","vehicleDestroys","weaponsAcquired","rideDistance","walkDistance","swimDistance","winPlacePerc"]

    length = len(res_dict[new_labels[1]])
    new_data_table = []
    index_labels = []
    flag = True
    for i in range(length):
        tmp = []
        for index in res_dict:
            if flag :
                index_labels.append(index)
            tmp.append(res_dict[index][i])
        flag = False
        new_data_table.append(tmp)
    
    if WH_D_L == "DICT":
        return res_dict, new_labels
    elif WH_D_L == "LIST":
        return new_data_table, new_labels, win_pre
    else :
        return None

name_set = ["solo","squad","duo","solo_fpp","duo_fpp","squad_fpp"]

for name in name_set:
    path_Open = "./train_"+name+"_V2.csv"
    path_Save = "./V6_data/train_"+name+"_V6.csv"
    F = open(path_Open,'r',encoding='utf-8')
    rdr = csv.reader(F)
    data_table = []
    for line in rdr:
        data_table.append(line)

    labels = data_table[0]
    del data_table[0]

    new_TABLE ,new_label , winplace  = PREPROCESSING(data_table, labels, WH_D_L = "LIST")

    res = pd.DataFrame(new_TABLE, columns=new_label)
    res.to_csv(path_Save, header=True, index=False)
    print(name,"Complete")



