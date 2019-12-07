import numpy as np
import csv
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

F = open("./test_V2.csv","r", encoding='utf-8')
rdr = csv.reader(F)
data_table = []
for line in rdr:
    data_table.append(line)
labels = data_table[0]
num_rows = len(data_table)
matchT = labels.index("matchType")

squad_fpp = [labels]
duo_fpp = [labels]
solo_fpp = [labels]
duo = [labels]
solo = [labels]
squad = [labels]

for i in range(1,num_rows):
    con = data_table[i][matchT]
    if con == "squad-fpp":
        squad_fpp.append(data_table[i])
    elif con == "duo-fpp":
        duo_fpp.append(data_table[i])
    elif con == "solo-fpp":
        solo_fpp.append(data_table[i])
    elif con == "duo":
        duo.append(data_table[i])
    elif con == "solo":
        solo.append(data_table[i])
    elif con == "squad":
        squad.append(data_table[i])

F.close()

new_table_index = ["squad_fpp","duo_fpp","solo_fpp","squad","duo","solo"]
new_table = []
new_table.append(squad_fpp)
new_table.append(duo_fpp)
new_table.append(solo_fpp)
new_table.append(squad)
new_table.append(duo)
new_table.append(solo)
 
i = 0

for con in new_table_index:
    file_name = "./test_" + con + "_V2.csv"
    with open(file_name,"w",encoding='utf-8',newline="") as F:
        wr = csv.writer(F)
        for line in new_table[i]:
            wr.writerow(line)
    print(file_name + " Complete!")
    i += 1

