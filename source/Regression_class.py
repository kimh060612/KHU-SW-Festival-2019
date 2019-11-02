import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR 
import csv
import pandas as pd
import scipy
import os
import PREPRO as FILEPRO

class REGRESSION:
    
    def __init__(self, file_path, encode='utf-8'):
        self.path = file_path
        self.res_dict = {}
        self.F = open(self.path, "r", encoding=encode)
        self.rdr = csv.reader(self.F)
        self.table = []
        for line in self.rdr:
            self.table.append(line)
        self.labels = self.table[0]
        del self.table[0]
        self.win_pre = []
        
    
    def preprocessing(self, table, label):
        self.res_dict = {}
        for con in label:
            if con == "Id" or con == "groupId" or con == "matchId" or con == "matchType" or con == "winPlacePerc":
                continue
            elif con == "rideDistance" or con == "swimDistance" or con == "walkDistance":
                continue
            else :
                self.res_dict[con] = []
        self.res_dict["Distance"] = []
        for con in table:
            Distance = 0
            for index in label:
                if index == "Id" or index == "groupId" or index == "matchId" or index == "matchType":
                    continue
                elif index == "winPlacePerc":
                    try:
                        self.win_pre.append(float(con[label.index(index)]))
                    except:
                        self.win_pre.append(0)
                elif index == "rideDistance" or index == "swimDistance" or index == "walkDistance":
                    Distance += float(con[label.index(index)])
                else :
                    try:
                        self.res_dict[index].append(int(con[label.index(index)]))
                    except:
                        self.res_dict[index].append(float(con[label.index(index)]))
            self.res_dict["Distance"].append(Distance)
        self.new_labels = ["assists","boosts","damageDealt","DBNOs","headshotKills","heals","killPlace","killPoints","kills","killStreaks","longestKill","matchDuration","numGroups","rankPoints","revives","Distance","roadKills","teamKills","vehicleDestroys","weaponsAcquired","winPoints"]

    def print_classes(self):
        print(self.new_labels)
    
    def win_place_perc_mean(self):
        return np.mean(np.array(self.win_pre))
    
    def Plotting(self, X, Y):
        
        X_table = self.res_dict[X]
        Y_table = self.res_dict[Y]
        plt.scatter(X_table,Y_table,marker="o", color="red")
        plt.xlabel(X)
        plt.ylabel(Y)
        plt.show()

    # X is string that indicate the name of class 
    def Reg_Random_Forest(self, X, Y):
        
        X_table = np.array(self.res_dict[X])
        X_table = X_table.reshape(len(X_table),1)
        if Y == "winPlacePerc":
            Y_table = np.array(self.win_pre)
        else :
            Y_table = np.array(self.res_dict[Y])
        Y_table = Y_table.reshape(len(Y_table),1)

        xTrain, xTest, yTrain, yTest = train_test_split(X_table,Y_table, test_size=0.2, random_state=0)

        nTree_List = range(50, 500, 10)
        MESF = []
        for N in nTree_List:
            Regressor = RandomForestRegressor(n_estimators = N, random_state = 0)
            Regressor.fit(xTrain,yTrain)

            prediction = Regressor.predict(xTest)
            MESF.append(mean_squared_error(yTest, prediction))

        Max_Ntree = 50 + (MESF.index(min(MESF)))*10
        Regressor = RandomForestRegressor(n_estimators = Max_Ntree, random_state = 0)
        Regressor.fit(xTrain,yTrain)

        pred = Regressor.predict(xTest)

        X_grid = np.array(xTest)
        X_grid = X_grid.reshape((len(xTest),1))

        plt.scatter(xTrain, yTrain, color="red")
        plt.plot(X_grid,pred, color="blue")
        plt.title("R_F by " + X + " prediction")
        plt.xlabel(X)
        plt.ylabel(Y)

        plt.show()
    
    def Total_REG_RF(self, Importance_name):

        length = len(self.res_dict[self.new_labels[1]])
        new_data_table = []
        index_labels = []
        flag = True
        for i in range(length):
            tmp = []
            for index in self.res_dict:
                if flag :
                    index_labels.append(index)
                tmp.append(self.res_dict[index][i])
            flag = False
            new_data_table.append(tmp)
        
        new_data_table = np.array(new_data_table)
        Y_perc = np.array(self.win_pre)

        xTrain, xTest, yTrain, yTest = train_test_split(new_data_table, Y_perc, test_size=0.2, random_state=531)

        nTree_List = range(50, 200, 10)
        MESF = []
        for N in nTree_List:
            REGRESSOR = RandomForestRegressor(n_estimators=N, random_state=0)
            REGRESSOR.fit(xTrain, yTrain)
            prediction = REGRESSOR.predict(xTest)
            MESF.append(mean_squared_error(yTest, prediction))
            print(str(N) + "th ensemble tree complete!")

        minError = min(MESF)
        print(minError)
        Min_E_NTree = 50 + 10*MESF.index(minError)
        
        REGRESSOR = RandomForestRegressor(n_estimators=Min_E_NTree, random_state=0)
        REGRESSOR.fit(xTrain, yTrain)

        self.REG = REGRESSOR

        Var_Importance = REGRESSOR.feature_importances_
        Var_Importance = Var_Importance/Var_Importance.max()

        STR = ""
        for name, Importance in zip(index_labels, Var_Importance):
            STR += str(name) + " : " + str(Importance) + "\n"
        
        F = open("./"+Importance_name+".txt", "w", encoding="utf-8")
        F.write(STR)
        F.close()

    
    def PREDICTION(self, DATA_TABLE_NAME):
        table, labels = FILEPRO.INPUT(DATA_TABLE_NAME)

        X_table, labels, Y_table = FILEPRO.PREPROCESSING(table, labels, WH_D_L="LIST")

        X_table = np.array(X_table)
        Y_table = np.array(Y_table)
        labels = np.array(labels)

        pred = self.REG.predict(X_table)



        return pred



        
        
    

    
'''

A = REGRESSION("./V4_data/train_solo_V4.csv")
A.preprocessing(A.table,A.labels)

A.Total_REG_RF("Importance")

'''

'''
DATA_list = os.listdir("./V4_data")

print(DATA_list)
'''

''' match type에 따른 승률 분석
win_list = []

for con in DATA_list:
    A = REGRESSION("./V4_data/"+con)
    A.preprocessing(A.table,A.labels)
    
    win_list.append(A.win_place_perc_mean())
    print(A.win_place_perc_mean())

'''





    



