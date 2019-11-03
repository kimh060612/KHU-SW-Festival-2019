import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import minmax_scale, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import catboost as cat
from sklearn.metrics import mean_squared_error
import shap
import csv
import pandas as pd
import scipy
import os
import pickle
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

        self.new_labels = ["assists","boosts","damageDealt","DBNOs","headshotKills","heals","killPlace","killPoints","kills","killStreaks","longestKill","matchDuration","numGroups", "rankPoints","revives","roadKills","teamKills","vehicleDestroys","weaponsAcquired","winPoints","Distance"]

        self.DATA_TABLE = np.zeros((len(table),len(self.new_labels)))

        for i in range(len(table)):
            Distance = 0
            for index in label:
                if index == "Id" or index == "groupId" or index == "matchId" or index == "matchType" or index == "maxPlace":
                    continue
                elif index == "winPlacePerc":
                    try:
                        self.win_pre.append(float(table[i][label.index(index)]))
                    except:
                        self.win_pre.append(0)
                elif index == "rideDistance" or index == "swimDistance" or index == "walkDistance":
                    Distance += float(table[i][label.index(index)])
                else :
                    try :
                        self.DATA_TABLE[i][self.new_labels.index(index)] = int(table[i][label.index(index)])
                    except :
                        self.DATA_TABLE[i][self.new_labels.index(index)] = float(table[i][label.index(index)])
            if self.DATA_TABLE[i][self.new_labels.index("Distance")] == 0:
                self.DATA_TABLE[i][self.new_labels.index("Distance")] = Distance
        self.res_dict = pd.DataFrame(self.DATA_TABLE, columns=self.new_labels)

    def win_place_perc_mean(self):
        return np.mean(np.array(self.win_pre))
    
    def Plotting_scatter(self, X, Y):
        
        X_table = np.array(self.res_dict[X])
        if Y == "winPlacePerc":
            Y_table = np.array(self.win_pre)
        else :
            Y_table = np.array(self.res_dict[Y])
        plt.scatter(X_table,Y_table,marker="o", color="red")
        plt.xlabel(X)
        plt.ylabel(Y)
        plt.show()

    def Plotting_bar(self, X, Y):
        X_table = np.array(self.res_dict[X])
        if Y == "winPlacePerc":
            Y_table = np.array(self.win_pre)
        else :
            Y_table = np.array(self.res_dict[Y])
        
        plt.bar(X_table,Y_table)
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
    
    def Total_REG_RF(self, Importance_name, Importance_analye = True):

        X_table = np.array(self.DATA_TABLE)
        Y_perc = np.array(self.win_pre)

        xTrain, xTest, yTrain, yTest = train_test_split(X_table, Y_perc, test_size=0.2, random_state=531)

        REGRESSOR = RandomForestRegressor(n_estimators=60, random_state=0)
        REGRESSOR.fit(xTrain, yTrain)
        prediction = REGRESSOR.predict(xTest)
        mean_squared_error(yTest, prediction)
        self.REGRF = REGRESSOR

        if Importance_analye :
            Var_Importance = self.REGRF.feature_importances_
            Var_Importance = Var_Importance/Var_Importance.max()

            STR = ""
            for name, Importance in zip(self.new_labels, Var_Importance):
                STR += str(name) + " : " + str(Importance) + "\n"
            
            F = open("./"+Importance_name+".txt", "w", encoding="utf-8")
            F.write(STR)
            F.close()
        
        score = cross_val_score(self.REGRF, xTrain, yTrain, scoring="neg_mean_squared_error", cv=10)
        tree_rmse_score = np.sqrt(-score)
        print("score: ",np.mean(tree_rmse_score))


    def Total_REG_LG(self, Importance_name,  Importance_analye = True):
        
        X_data = np.array(self.DATA_TABLE)
        Y_perc = np.array(self.win_pre)
        #xTrain, xTest, yTrain, yTest = train_test_split(X_data, Y_perc, test_size=0.2, random_state=531)

        params = {'learning_rate': 0.01, 'max_depth': 16, 'boosting': 'gbdt', 'objective': 'regression', 'metric': 'rmse', 'num_leaves': 144, 'feature_fraction': 0.9, 'bagging_fraction': 0.7, 'bagging_freq': 5, 'seed':2018}
        
        pred = np.zeros(len(X_data))
        cv = KFold(n_splits=3,shuffle=True,random_state=0)
        Error = []
        for train_index, test_index in cv.split(X_data):
            xTrain, xTest = X_data[train_index], X_data[test_index]
            yTrain, yTest = Y_perc[train_index], Y_perc[test_index]

            train_ds = lgb.Dataset(xTrain, label=yTrain)
            val_ds = lgb.Dataset(xTest, label=yTest)

            self.REGLG = lgb.train(params, train_ds, 1000, val_ds,verbose_eval=10, early_stopping_rounds=100)
            pred[test_index] = self.REGLG.predict(xTest)
            Error.append(mean_squared_error(pred[test_index],yTest))
        lgb.plot_importance(self.REGLG)
        pl.title(Importance_name)
        pl.show()
        LG_rmse_score = np.sqrt(np.mean(Error))
        print("score: ",LG_rmse_score)

    def Total_REG_XG(self, Importance_name, Importance_analye = True):
        
        X_data = np.array(self.DATA_TABLE)
        Y_perc = np.array(self.win_pre)

        params = {'objective': 'reg:linear', 'eval_metric': 'rmse', 'eta': 0.005, 'max_depth': 15, 'subsample': 0.6, 'colsample_bytree': 0.6, 'alpha':0.001, 'random_state': 42, 'silent': True}

        pred = np.zeros(len(X_data))
        cv = KFold(n_splits=3,shuffle=True,random_state=0)
        Error = []
        
        for train_index, test_index in cv.split(X_data):
            xTrain, xTest = X_data[train_index], X_data[test_index]
            yTrain, yTest = Y_perc[train_index], Y_perc[test_index]

            train_ds = xgb.DMatrix(xTrain, yTrain)
            val_ds = xgb.DMatrix(xTest, yTest)
            watch_list = [(train_ds, "TRAIN"),(val_ds, "VAL")]

            self.REGXG = xgb.train(params, train_ds, 2100, watch_list, maximize=False, early_stopping_rounds = 100, verbose_eval=100)
            
            pred[test_index] = self.REGXG.predict(xgb.DMatrix(xTest))
            Error.append(mean_squared_error(pred[test_index], yTest))
        xgb.plot_importance(self.REGXG)
        pl.title(Importance_name)
        pl.show()
        XG_rmse_score = np.sqrt(np.mean(Error))
        print("score: ",XG_rmse_score)

    def Total_REG_CAT(self, Importance_name, Importance_analye = True):
        
        X_data = np.array(self.DATA_TABLE)
        Y_perc = np.array(self.win_pre)

        pred = np.zeros(len(X_data))
        cv = KFold(n_splits=3,shuffle=True,random_state=0)
        Error = []

        for train_index, test_index in cv.split(X_data):
            xTrain, xTest = X_data[train_index], X_data[test_index]
            yTrain, yTest = Y_perc[train_index], Y_perc[test_index]

            self.REGCAT = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=4, l2_leaf_reg=20, bootstrap_type='Bernoulli', subsample=0.6, eval_metric='RMSE', metric_period=50, od_type='Iter', od_wait=45, random_seed=17, allow_writing_files=False)
            self.REGCAT.fit(xTrain, yTrain, eval_set=(xTest, yTest), use_best_model=True, verbose=True)

            pred[test_index] = self.REGCAT.predict(xTest)
            Error.append(mean_squared_error(pred[test_index],yTest))
        
        CAT_rmse_score = np.sqrt(np.mean(Error))
        print("score: ",CAT_rmse_score)

    # RF = random forest,LG = lightGBM,XG = XGboost
    def PREDICTION(self, DATA_TABLE_NAME,REGRESSOR_NAME = "RF"):
        table, labels = FILEPRO.INPUT(DATA_TABLE_NAME)

        X_table, labels, Y_table = FILEPRO.PREPROCESSING(table, labels, WH_D_L="LIST")

        X_table = np.array(X_table)
        Y_table = np.array(Y_table)
        labels = np.array(labels)

        if REGRESSOR_NAME == "RF":
            pred = self.REGRF.predict(X_table)
        elif REGRESSOR_NAME == "LG":
            pred = self.REGLG.predict(X_table)
        elif REGRESSOR_NAME == "XG":
            pred = self.REGXG.predict(X_table)
        elif REGRESSOR_NAME == "CAT":
            pred = self.REGCAT.predict(X_table)
        '''
        with open("./" + DATA_TABLE_NAME + "_result" + ".csv", "w", encoding='utf-8') as FILE:
            FILE.writelines(pred)
        print(pred)
        '''
        return pred
    
    def SHAP_Analysis(self, REGRESSOR_NAME = "RF", dependence_plot_name = []):
        
        shap.initjs()

        DATA_TABLE = self.res_dict

        if REGRESSOR_NAME == "XG":
            explainer = shap.TreeExplainer(self.REGXG)
            shap_val = explainer.shap_values(DATA_TABLE)
            shap.summary_plot(shap_val, DATA_TABLE, plot_type="bar",max_display=21)
        elif REGRESSOR_NAME == "LG":
            explainer = shap.TreeExplainer(self.REGLG)
            shap_val = explainer.shap_values(DATA_TABLE)
            shap.summary_plot(shap_val, DATA_TABLE, plot_type="bar",max_display=21)
            shap.summary_plot(shap_val,DATA_TABLE, plot_type="dot",max_display=21)
            for index in dependence_plot_name :
                shap.dependence_plot(index, shap_val, DATA_TABLE)
        elif REGRESSOR_NAME == "CAT":
            explainer = shap.TreeExplainer(self.REGCAT)
            shap_val = explainer.shap_values(DATA_TABLE)
            shap.summary_plot(shap_val, DATA_TABLE, plot_type="bar", max_display=21)
            shap.summary_plot(shap_val,DATA_TABLE, plot_type="dot",max_display=21)
            for index in dependence_plot_name :
                shap.dependence_plot(index, shap_val, DATA_TABLE)
        elif REGRESSOR_NAME == "RF":
            explainer = shap.TreeExplainer(self.REGRF)
            shap_val = explainer.shap_values(DATA_TABLE)
            shap.summary_plot(shap_val, DATA_TABLE, plot_type="bar",max_display=21)
            shap.summary_plot(shap_val,DATA_TABLE, plot_type="dot",max_display=21)
            for index in dependence_plot_name :
                shap.dependence_plot(index, shap_val, DATA_TABLE)

    

A = REGRESSION("./V4_data/train_duo_V4.csv")
A.preprocessing(A.table,A.labels)

A.Total_REG_CAT("Importance")
A.SHAP_Analysis("CAT",["kills"])


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





    



