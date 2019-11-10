import numpy as np
import csv

def Check_num_Kill(table, IK,IHK,IRK):
    for con in table:
        if int(con[IK]) < int(con[IHK]):
            return False
        elif int(con[IK]) < int(con[IRK]):
            return False
        else :
            continue
    return True

def check_Kill_streak(table, IK, IKS):
    for con in table:
        if int(con[IK]) < int(con[IKS]):
            return False
        else :
            continue
    return True


F = open("./train_V2.csv","r", encoding='utf-8')
rdr = csv.reader(F)
data_table = []
for line in rdr:
    data_table.append(line)
labels = data_table[0]
del data_table[0]
num_rows = len(data_table)
index_kill = labels.index("kills")
index_headShot = labels.index("headshotKills")
index_roadkill = labels.index("roadKills")
index_KillStreak = labels.index("killStreaks")

print(Check_num_Kill(data_table, index_kill,index_headShot,index_roadkill))
print(check_Kill_streak(data_table, index_kill,index_KillStreak))
