import csv
import numpy as np
import pandas as pd

def INPUT(file_name):
    F = open("./"+file_name+".csv", "r", encoding="utf-8")
    rdr = csv.reader(F)
    table = []
    for con in rdr:
        table.append(con)
    labels = table[0]
    del table[0]

    return table, labels

def PREPROCESSING(table, labels, WH_D_L = "DICT"):

    new_labels = ["assists","boosts","damageDealt","DBNOs","headshotKills","heals","killPlace","killPoints","kills","killStreaks","longestKill","matchDuration","numGroups", "rankPoints","revives","Distance","roadKills","teamKills","vehicleDestroys","weaponsAcquired","winPoints"]

    DATA_TABLE = np.zeros((len(table),len(new_labels)))

    win_place_perc = []

    for i in range(len(table)):
        Distance = 0
        for index in labels:
            if index == "Id" or index == "groupId" or index == "matchId" or index == "matchType" or index == "maxPlace":
                continue
            elif index == "winPlacePerc":
                try:
                    win_place_perc.append(float(table[i][labels.index(index)]))
                except:
                    win_place_perc.append(0)
            elif index == "rideDistance" or index == "swimDistance" or index == "walkDistance":
                Distance += float(table[i][labels.index(index)])
            else :
                try :
                    DATA_TABLE[i][new_labels.index(index)] = int(table[i][labels.index(index)])
                except :
                    DATA_TABLE[i][new_labels.index(index)] = float(table[i][labels.index(index)])
            if DATA_TABLE[i][new_labels.index("Distance")] == 0:
                DATA_TABLE[i][new_labels.index("Distance")] = Distance
        res_dict = pd.DataFrame(DATA_TABLE, columns=new_labels)
    
    if WH_D_L == "DICT":
        return res_dict, new_labels
    elif WH_D_L == "LIST":
        return DATA_TABLE, new_labels, win_place_perc
    else :
        return None


