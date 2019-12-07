import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

'''
Doing exactly same data preprocessing in the data we want to predict winplaceperc.
This module is used in the REGRESSION_CLASS_RE.py, PREDICT method.
'''

def INPUT(file_name):
    F = open("./"+file_name+".csv", "r", encoding="utf-8")
    rdr = csv.reader(F)
    table = []
    for con in rdr:
        table.append(con)
    labels = table[0]
    del table[0]

    return table, labels

def PREPROCESSING(table, labels, WH_D_L = "DICT",scale = False, scaler="minmax"):

    res_dict = {}
    for con in labels:
        if con == "Id" or con == "groupId" or con == "matchId" or con == "matchType" or con == "winPlacePerc" or con == "maxPlace" or con == "rankPoints" or con == "killPoints" or con == "winPoints":
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
                    win_pre.append(float(con[labels.index(index)]))
                except:
                    win_pre.append(0)
            else :
                try:
                    res_dict[index].append(int(con[labels.index(index)]))
                except:
                    res_dict[index].append(float(con[labels.index(index)]))
    new_labels = ["assists","boosts","damageDealt","DBNOs","headshotKills","heals","killPlace","kills","killStreaks","longestKill","matchDuration","numGroups","revives","roadKills","teamKills","vehicleDestroys","weaponsAcquired","rideDistance","walkDistance","swimDistance"]

    if scale :
        if scaler == "minmax":
                Sca = MinMaxScaler()
                for index in res_dict:
                    X = np.array(res_dict[index])
                    X = X.reshape(-1,1)
                    Sca.fit(X)
                    X_scale = Sca.transform(X)
                    res_dict[index] = X_scale.reshape(1,-1).tolist()[0]
        elif scaler == "Standard":
                Sca = StandardScaler()
                for index in res_dict:
                    X = np.array(res_dict[index])
                    X = X.reshape(-1,1)
                    Sca.fit(X)
                    X_scale = Sca.transform(X)
                    res_dict[index] = X_scale.reshape(1,-1).tolist()[0]
        elif scaler == "Ro" :
                Sca = RobustScaler()
                for index in res_dict:
                    X = np.array(res_dict[index])
                    X = X.reshape(-1,1)
                    Sca.fit(X)
                    X_scale = Sca.transform(X)
                    res_dict[index] = X_scale.reshape(1,-1).tolist()[0]
        else :
                print("Error")

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
        return new_data_table, index_labels, win_pre
    else :
        return None


