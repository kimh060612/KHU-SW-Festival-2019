import csv

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

    res_dict = {}
    win_place_perc = []
    for con in labels:
        if con == "Id" or con == "groupId" or con == "matchId" or con == "matchType" or con == "winPlacePerc":
            continue
        elif con == "rideDistance" or con == "swimDistance" or con == "walkDistance":
            continue
        else :
            res_dict[con] = []
    res_dict["Distance"] = []

    for con in table:
        Distance = 0
        for index in labels:
            if index == "Id" or index == "groupId" or index == "matchId" or index == "matchType":
                continue
            elif index == "winPlacePerc":
                try:
                    win_place_perc.append(float(con[label.index(index)]))
                except:
                    win_place_perc.append(0)
            elif index == "rideDistance" or index == "swimDistance" or index == "walkDistance":
                Distance += float(con[labels.index(index)])
            else :
                try:
                    res_dict[index].append(int(con[label.index(index)]))
                except:
                    res_dict[index].append(float(con[labels.index(index)]))
        res_dict["Distance"].append(Distance)
    new_labels = ["assists","boosts","damageDealt","DBNOs","headshotKills","heals","killPlace","killPoints","kills","killStreaks","longestKill","matchDuration","numGroups","rankPoints","revives","Distance","roadKills","teamKills","vehicleDestroys","weaponsAcquired","winPoints"]

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
        return new_data_table, index_labels, win_place_perc
    else :
        return None


