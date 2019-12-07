import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import seaborn as sns
import math

#Not used

def Grouping_Var(Data):
    labeled = []
    mean = DATA.mean()
    std = DATA.std()
    Max = max(DATA)
    Min = min(DATA)
    First_point = math.ceil(mean - 1.5*std)
    Third_point = math.ceil(mean + 1.5*std)
    for i in range(len(DATA)):
        Z = (DATA[i] - mean)/std
        if Z < -1.5 :
            labeled.append(str(Min) + "~" + str(First_point))
        elif -1.5 < Z and Z < 0:
            labeled.append(str(First_point) + "~" + str(mean))
        elif Z > 0 and Z < 1.5 :
            labeled.append(str(mean) + "~" + str(Third_point))
        elif Z > 1.5 :
            labeled.append(str(Third_point) + "~" + str(Max))
    return labeled

File_name = ["solo_fpp","duo_fpp","squad_fpp","solo","squad","duo"]

label = []
Table = []

for i in range(len(File_name)):
    tmp = []
    path = "./train_" + File_name[i] + "_V2.csv"
    F = open(path, "r", encoding="utf-8")
    rdr = csv.reader(F)
    for line in rdr:
        tmp.append(line)
    label = tmp[0]
    del tmp[0]
    for line in tmp:
        Table.append(line)

DATA = pd.DataFrame(Table, columns=label)
#DATA.to_csv("./New_train_V2.csv", header=True, index= False)

Valid_Var = ['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'heals', 'kills', 'killStreaks', 'longestKill', 'matchDuration', 'numGroups', 'revives', 'rideDistance', 'roadKills', 'swimDistance', 'teamKills', 'vehicleDestroys', 'weaponsAcquired', 'winPlacePerc']
Small_Valid_Var = []
Big_Valid_Var = []

for con in Valid_Var :
    if DATA[con].max() > 1000:
        Big_Valid_Var.append(con)
    else:
        Small_Valid_Var.append(con)

