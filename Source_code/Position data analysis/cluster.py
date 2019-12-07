from sklearn.cluster import k_means
import numpy
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

"""
Analyze the kill-killed position data
"""

F = open("kill_match_stats_final_0.csv","r",encoding='utf-8')
rdr = csv.reader(F)
data_table = []
for lines in rdr:
    data_table.append(lines)

labels = data_table[0]
del data_table[0]

TABLE = pd.DataFrame(data_table, columns=labels)



MIRANMAR_V_TABLE = {}
MIRANMAR_V_TABLE["X"] = []
MIRANMAR_V_TABLE["Y"] = []
MIRANMAR_K_TABLE = {}
MIRANMAR_K_TABLE["X"] = []
MIRANMAR_K_TABLE["Y"] = []

ERANGEL_V_TABLE = {}
ERANGEL_V_TABLE["X"] = []
ERANGEL_V_TABLE["Y"] = []
ERANGEL_K_TABLE = {}
ERANGEL_K_TABLE["X"] = []
ERANGEL_K_TABLE["Y"] = []

total_num_Miran = 0
total_num_Eran = 0

for row in data_table:
    if row[labels.index("killer_name")] == row[labels.index("victim_name")]:
        continue
    try:
        if row[labels.index("map")] == "MIRAMAR":
            total_num_Miran += 1
            MIRANMAR_K_TABLE["X"].append(float(row[labels.index("killer_position_x")]))
            MIRANMAR_K_TABLE["Y"].append(float(row[labels.index("killer_position_y")]))
            MIRANMAR_V_TABLE["X"].append(float(row[labels.index("victim_position_x")]))
            MIRANMAR_V_TABLE["Y"].append(float(row[labels.index("victim_position_y")]))
            
        elif row[labels.index("map")] == "ERANGEL":
            total_num_Eran += 1
            ERANGEL_K_TABLE["X"].append(float(row[labels.index("killer_position_x")]))
            ERANGEL_K_TABLE["Y"].append(float(row[labels.index("killer_position_y")]))
            ERANGEL_V_TABLE["X"].append(float(row[labels.index("victim_position_x")]))
            ERANGEL_V_TABLE["Y"].append(float(row[labels.index("victim_position_y")]))
    except:
        continue        

Sniping_Miran = {}
Sniping_Miran["kill_X"] = []
Sniping_Miran["kill_Y"] = []
Sniping_Miran["Vic_X"] = []
Sniping_Miran["Vic_Y"] = []
Sniping_Eran = {}
Sniping_Eran["kill_X"] = []
Sniping_Eran["kill_Y"] = []
Sniping_Eran["Vic_X"] = []
Sniping_Eran["Vic_Y"] = []

for i in range(total_num_Miran):
    distance = math.sqrt((MIRANMAR_K_TABLE["X"][i] - MIRANMAR_V_TABLE["X"][i])*(MIRANMAR_K_TABLE["X"][i] - MIRANMAR_V_TABLE["X"][i]) + (MIRANMAR_K_TABLE["Y"][i] - MIRANMAR_V_TABLE["Y"][i])*(MIRANMAR_K_TABLE["Y"][i] - MIRANMAR_V_TABLE["Y"][i]))
    if distance >= 1000:
        Sniping_Miran["kill_X"].append(MIRANMAR_K_TABLE["X"][i]) 
        Sniping_Miran["kill_Y"].append(MIRANMAR_K_TABLE["Y"][i]) 
        Sniping_Miran["Vic_X"].append(MIRANMAR_V_TABLE["X"][i]) 
        Sniping_Miran["Vic_Y"].append(MIRANMAR_V_TABLE["Y"][i])
    
plt.figure("M", figsize=(1200,1200))
plt.scatter(MIRANMAR_K_TABLE["X"],MIRANMAR_K_TABLE["Y"], color="red")
plt.scatter(MIRANMAR_V_TABLE["X"],MIRANMAR_V_TABLE["Y"], color="blue")
plt.show()

plt.figure("E", figsize=(12000,1200))
plt.scatter(ERANGEL_K_TABLE["X"],ERANGEL_K_TABLE["Y"], color="red")
plt.scatter(ERANGEL_V_TABLE["X"],ERANGEL_V_TABLE["Y"], color="blue")
plt.show()
