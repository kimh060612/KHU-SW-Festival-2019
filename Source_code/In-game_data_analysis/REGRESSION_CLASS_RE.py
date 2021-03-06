import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
import catboost
from catboost import CatBoostRegressor
import catboost as cat
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import shap
import csv
import pandas as pd
import scipy
import os
import pickle
import PREPRO as FILEPRO
from DeepRegressor import DeepRegression

'''
This source is main source of our data analysis, regressoin for data and many other.
The class REFRESSION can be classified into four parts: data preprocessing, plotting & Regression for One Variable respect to another, Regression and predict for winplaceperc, and model interpretation.
Each of these parts will be explained in detail in the code below.
This code is built for managing several experiment easily
'''

def plot_history(hist, Save_name):

    plt.figure(figsize=(8,12))
    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],label='Train Error')
    plt.ylim([0.04,0.08])
    plt.legend()

    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'Val Error')
    plt.ylim([0.04,0.08])
    plt.legend()

    plt.savefig("./" + Save_name + ".png")

    epoch = hist['epoch']
    mae = hist['val_mean_absolute_error']

    for i in range(len(epoch) - 10,len(epoch)):
        print("epoch: ", epoch[i] ,"  mae: ", mae[i])


class REGRESSION:
    
    # getting the data file
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

    #Preprocessing part for data. exclude Exception and useless variable, Scaling data(optional), and reconstructing the data table. 
    def preprocessing(self, table, label, scale = False, scaler = "minmax", OUTD = False):
        self.res_dict = {}
        for con in label:
            if con == "Id" or con == "groupId" or con == "matchId" or con == "matchType" or con == "winPlacePerc" or con == "maxPlace" or con == "rankPoints" or con == "killPoints" or con == "winPoints":
                continue
            else :
                self.res_dict[con] = []
        for con in table:
            for index in label:
                if index == "Id" or index == "groupId" or index == "matchId" or index == "matchType" or index == "maxPlace" or index == "rankPoints" or index == "killPoints" or index == "winPoints":
                    continue
                elif index == "winPlacePerc":
                    try:
                        self.win_pre.append(float(con[label.index(index)]))
                    except:
                        self.win_pre.append(0)
                else :
                    try:
                        self.res_dict[index].append(int(con[label.index(index)]))
                    except:
                        self.res_dict[index].append(float(con[label.index(index)]))
        self.new_labels = ["assists","boosts","damageDealt","DBNOs","headshotKills","heals","killPlace","kills","killStreaks","longestKill","matchDuration","numGroups","revives","roadKills","teamKills","vehicleDestroys","weaponsAcquired","rideDistance","walkDistance","swimDistance"]
        
        if not OUTD:
            
            pass
        
        if scale :
            if scaler == "minmax":
                Sca = MinMaxScaler()
                for index in self.res_dict:
                    X = np.array(self.res_dict[index])
                    X = X.reshape(-1,1)
                    Sca.fit(X)
                    X_scale = Sca.transform(X)
                    self.res_dict[index] = X_scale.reshape(1,-1).tolist()[0]
            elif scaler == "Standard":
                Sca = StandardScaler()
                for index in self.res_dict:
                    X = np.array(self.res_dict[index])
                    X = X.reshape(-1,1)
                    Sca.fit(X)
                    X_scale = Sca.transform(X)
                    self.res_dict[index] = X_scale.reshape(1,-1).tolist()[0]
            elif scaler == "Ro" :
                Sca = RobustScaler()
                for index in self.res_dict:
                    X = np.array(self.res_dict[index])
                    X = X.reshape(-1,1)
                    Sca.fit(X)
                    X_scale = Sca.transform(X)
                    self.res_dict[index] = X_scale.reshape(1,-1).tolist()[0]
            else :
                print("Error")

        length = len(self.res_dict[self.new_labels[1]])
        self.new_data_table = []
        self.index_labels = []
        flag = True
        for i in range(length):
            tmp = []
            for index in self.res_dict:
                if flag :
                    self.index_labels.append(index)
                tmp.append(self.res_dict[index][i])
            flag = False
            self.new_data_table.append(tmp)

        '''
        numData = len(self.new_data_table)
        X_data = np.array(self.new_data_table)
        Y_data = np.array(self.win_pre)
        Label = np.array(self.index_labels)
        Xtable = pd.DataFrame(X_data, columns = self.index_labels)
        abnormalList = []
        for i in range(numData):
            if Xtable["walkDistance"][i] < 0.5 and Y_data[i] > 0.98:
                abnormalList.append(i)
            elif Xtable["kills"][i] > 45 :
                abnormalList.append(i)
            elif Xtable["weaponsAcquired"][i] > 60 :
                abnormalList.append(i)
            elif Xtable["kills"][i] > 10 :
                if (Xtable["headshotKills"][i] / Xtable["kills"][i]) > 0.95 :
                    abnormalList.append(i)
            elif Xtable["walkDistance"][i] > 5000 or Xtable["rideDistance"][i] > 10000:
                abnormalList.append(i)    
        for i in range(len(abnormalList)):
            del self.new_data_table[i]
            del self.win_pre[i]
        '''

    def win_place_perc_mean(self):
        return np.mean(np.array(self.win_pre))
    
    # Plotting and Regression the data 
    def Plotting_scatter(self, X, Y):
        
        X_table = self.res_dict[X]
        if Y == "winPlacePerc":
            Y_table = self.win_pre
        else :
            Y_table = self.res_dict[Y]
        plt.scatter(X_table,Y_table,marker="o", color="red")
        plt.xlabel(X)
        plt.ylabel(Y)
        plt.show()

    def Plotting_bar(self, X, Y):
        X_table = self.res_dict[X]
        if Y == "winPlacePerc":
            Y_table = self.win_pre
        else :
            Y_table = self.res_dict[Y]
        
        plt.bar(X_table,Y_table)
        plt.xlabel(X)
        plt.ylabel(Y)
        plt.show()


    # Regression for the One variable respect to another.
    # X is string that indicate the name of class 
    # Y is the same context for X
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
    
    # The four methods following below is the regression part for winplaceperc respect to all other variable. 
    # Each of these methods varies their regressoin algorithms.
    # RF = random forest, XG = XGBoost, LG = LightGBM, CAT = CatBoost
    def Total_REG_RF(self, Importance_name="Importance", Importance_analye = True):

        X_table = np.array(self.new_data_table)
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
            for name, Importance in zip(self.index_labels, Var_Importance):
                STR += str(name) + " : " + str(Importance) + "\n"
            
            F = open("./"+Importance_name+".txt", "w", encoding="utf-8")
            F.write(STR)
            F.close()
        
        score = cross_val_score(self.REGRF, xTrain, yTrain, scoring="neg_mean_squared_error", cv=10)
        tree_rmse_score = np.sqrt(-score)
        print("score: ",np.mean(tree_rmse_score))


    def Total_REG_LG(self, Importance_name="Importance",  Importance_analye = True, epoch = 1000):
        
        X_data = np.array(self.new_data_table)
        Y_perc = np.array(self.win_pre)
        #xTrain, xTest, yTrain, yTest = train_test_split(X_data, Y_perc, test_size=0.2, random_state=531)
        
        evals_res = {}
        # num leaves 조절하면서 학습 실험 잰행.. ==> 이번에는 정확도 위주로 
        params = {'learning_rate': 0.01, 'max_depth': 16, 'boosting': 'gbdt', 'objective': 'regression', 'metric': 'mean_absolute_error', 'is_training_metric': True, 'num_leaves': 72, 'feature_fraction': 0.9, 'bagging_fraction': 0.7, 'bagging_freq': 5, 'seed':2018, 'device' : 'gpu'}
        Val_pred = np.zeros(len(X_data))
        Train_pred = np.zeros(len(X_data))
        cv = KFold(n_splits=5,shuffle=True,random_state=0)
        Val_Error = []
        Train_Error = []
        Epoch = 0
        epoch_label = []

        for train_index, test_index in cv.split(X_data):
            xTrain, xTest = X_data[train_index], X_data[test_index]
            yTrain, yTest = Y_perc[train_index], Y_perc[test_index]

            train_ds = lgb.Dataset(xTrain, label=yTrain)
            val_ds = lgb.Dataset(xTest, label=yTest, reference= train_ds)

            self.REGLG = lgb.train(params, train_ds, epoch, valid_sets=[train_ds, val_ds], verbose_eval=10, early_stopping_rounds=100, evals_result = evals_res)
            #Epoch += 1
            #epoch_label.append(Epoch)
            #Val_pred[test_index] = self.REGLG.predict(xTest)
            #Train_pred[train_index] = self.REGLG.predict(xTrain)
            #Train_Error.append(mean_absolute_error(Train_pred[train_index], yTrain))
            #Val_Error.append(mean_absolute_error(Val_pred[test_index],yTest))
        
        #Hist = {}
        #Hist['epoch'] = epoch_label
        #Hist['mean_absolute_error'] = Train_Error
        #Hist['val_mean_absolute_error'] = Val_Error
        #plot_history(Hist, "LightGBM_History")

        ax = lgb.plot_metric(evals_res, xlabel="Epoch", ylabel="Mean Abs Error")
        pl.title("Mean absolute error")
        pl.show()
        pl.savefig("./LGHIST.png")
        #LG_rmse_score = np.sqrt(np.mean(Val_Error))
        #print("score: ",LG_rmse_score)
        self.REGLG.save_model('model.txt', num_iteration=self.REGLG.best_iteration)

    def Total_REG_XG(self, Importance_name="Importance", Importance_analye = True, epoch = 2100):
        
        X_data = np.array(self.new_data_table)
        Y_perc = np.array(self.win_pre)

        params = {'objective': 'reg:linear', 'eval_metric': 'rmse', 'eta': 0.005, 'max_depth': 30, 'subsample': 0.6, 'colsample_bytree': 0.6, 'alpha':0.001, 'random_state': 42, 'silent': True}
        params['gpu_id'] = 0
        params['tree_method'] = 'gpu_hist'

        pred = np.zeros(len(X_data))
        cv = KFold(n_splits=5,shuffle=True,random_state=0)
        Error = []
        
        for train_index, test_index in cv.split(X_data):
            xTrain, xTest = X_data[train_index], X_data[test_index]
            yTrain, yTest = Y_perc[train_index], Y_perc[test_index]

            train_ds = xgb.DMatrix(xTrain, yTrain)
            val_ds = xgb.DMatrix(xTest, yTest)
            watch_list = [(train_ds, "TRAIN"),(val_ds, "VAL")]

            self.REGXG = xgb.train(params, train_ds, epoch, watch_list, maximize=False, early_stopping_rounds = 100, verbose_eval=100)
            
            pred[test_index] = self.REGXG.predict(xgb.DMatrix(xTest))
            Error.append(mean_squared_error(pred[test_index], yTest))
        xgb.plot_importance(self.REGXG)
        pl.title(Importance_name)
        pl.show()
        XG_rmse_score = np.sqrt(np.mean(Error))
        print("score: ",XG_rmse_score)

    def Total_REG_CAT(self, Importance_name="Importance", Importance_analye = True, epoch = 1000):
        
        X_data = np.array(self.new_data_table)
        Y_perc = np.array(self.win_pre)

        Val_pred = np.zeros(len(X_data))
        Train_pred = np.zeros(len(X_data))
        cv = KFold(n_splits=5,shuffle=True,random_state=0)
        Val_Error = []
        Train_Error = []
        Epoch = 0
        epoch_label = []
        Path_train = "/data/PUBG_EDA/Source_code/In-game_data_analysis/"
        for train_index, test_index in cv.split(X_data):
            xTrain, xTest = X_data[train_index], X_data[test_index]
            yTrain, yTest = Y_perc[train_index], Y_perc[test_index]

            self.REGCAT = CatBoostRegressor(iterations=1000, learning_rate=0.1, depth=4, l2_leaf_reg=20, bootstrap_type='Bernoulli', subsample=0.6, eval_metric='MAE', metric_period=50, od_type='Iter', od_wait=45, random_seed=17, allow_writing_files=False, task_type="GPU", devices='0', train_dir=Path_train)
            self.REGCAT.fit(xTrain, yTrain, eval_set=(xTest, yTest), use_best_model=True, verbose=False, plot= True)
            
            Epoch += 1
            epoch_label.append(Epoch)
            Val_pred[test_index] = self.REGCAT.predict(xTest)
            Train_pred[train_index] = self.REGCAT.predict(xTrain)
            Train_Error.append(mean_absolute_error(Train_pred[train_index], yTrain))
            Val_Error.append(mean_absolute_error(Val_pred[test_index],yTest))
        
        #Hist = {}
        #Hist['epoch'] = epoch_label
        #Hist['mean_absolute_error'] = Train_Error
        #Hist['val_mean_absolute_error'] = Val_Error
        #plot_history(Hist, "CatBoost_History")
        #self.REGCAT.
        #print("R2 Score: ", self.REGCAT.score())
        CAT_rmse_score = np.sqrt(np.mean(Val_Error))
        print("score: ",CAT_rmse_score)
        self.REGCAT.save_model("./CATMODEL")
        w = catboost.MetricVisualizer(Path_train,subdirs=True)
        w.start()
    
    def Total_REG_Deep(self, epoch = 1000):
        
        X_data = np.array(self.new_data_table)
        Y_perc = np.array(self.win_pre)
        Data_Shape = X_data.shape[1]
        
        self.DeepREG = DeepRegression(Data_Shape, X_data,Y_perc)
        self.DeepREG.Train(epoch)
    

    def Total_REG_Ridge(self, alpha = 1.0):
        
        X_data = np.array(self.new_data_table)
        Y_perc = np.array(self.win_pre)
        
        Val_pred = np.zeros(len(X_data))
        Train_pred = np.zeros(len(X_data))
        cv = KFold(n_splits=5,shuffle=True,random_state=0)
        #Val_Error = []
        #Train_Error = []
        Epoch = 0
        #epoch_label = []
        train_sizes = None
        train_scores = None
        valid_scores = None

        for train_index, test_index in cv.split(X_data):
            xTrain, xTest = X_data[train_index], X_data[test_index]
            yTrain, yTest = Y_perc[train_index], Y_perc[test_index]
            
            self.Ridge = Ridge(alpha=alpha)
            self.Ridge.fit(xTrain, yTrain)
            
            print("Ridge Training Score: ", self.Ridge.score(xTrain, yTrain))
            print("Ridge Test Score: ", self.Ridge.score(xTest, yTest))

        #Hist = {}
        #Hist['epoch'] = epoch_label
        #Hist['mean_absolute_error'] = Train_Error
        #Hist['val_mean_absolute_error'] = Val_Error
        #plot_history(Hist, "Ridge_History")
        #print("Ridge MAE: ", np.mean(Val_Error))    
        

    # predict winplaceperc with the model we made
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
        elif REGRESSOR_NAME == "Deep":
            pred = self.DeepREG.Predict(X_table)

        #print(mean_squared_error(pred, Y_table))

        return pred
    
    # Interpretation of model we made with SHAP value
    # Each Algorithms have their own mdoel interpretation algorithms. But the confidence of the result shows that SHAP value has higher confidence than those experimently.
    def SHAP_Analysis(self, REGRESSOR_NAME = "RF", dependence_plot_name = [], random_sample_size = 1000):
        
        shap.initjs()
        X_data = np.array(self.new_data_table)
        Y_perc = np.array(self.win_pre)

        BackGround = X_data[np.random.choice(X_data.shape[0], random_sample_size, replace = False)]
        Back_Ground_Table = pd.DataFrame(BackGround, columns = self.index_labels)

        DATA_TABLE = pd.DataFrame(self.new_data_table, columns = self.index_labels)
        

        if REGRESSOR_NAME == "XG":
            explainer = shap.TreeExplainer(self.REGXG)
            shap_val = explainer.shap_values(Back_Ground_Table)
            shap.summary_plot(shap_val, Back_Ground_Table, plot_type="bar",max_display=21)
        
        elif REGRESSOR_NAME == "RI":
            explainer = shap.KernelExplainer(self.Ridge.predict, BackGround)
            shap_val = explainer.shap_values(BackGround)
            shap.summary_plot(shap_val, Back_Ground_Table, plot_type="bar", max_display = 21)
            shap.summary_plot(shap_val, Back_Ground_Table, plot_type="dot", max_display = 21)
            for index in dependence_plot_name:
                shap.dependence_plot(index, shap_val, Back_Ground_Table)

        elif REGRESSOR_NAME == "LG":
            explainer = shap.TreeExplainer(self.REGLG)
            shap_val = explainer.shap_values(BackGround)
            shap.summary_plot(shap_val, Back_Ground_Table, plot_type="bar",max_display=21)
            shap.summary_plot(shap_val,Back_Ground_Table, plot_type="dot",max_display=21)
            for index in dependence_plot_name :
                shap.dependence_plot(index, shap_val, Back_Ground_Table)
        
        elif REGRESSOR_NAME == "CAT":
            explainer = shap.TreeExplainer(self.REGCAT)
            shap_val = explainer.shap_values(BackGround)
            shap.summary_plot(shap_val, Back_Ground_Table, plot_type="bar", max_display=21)
            shap.summary_plot(shap_val,Back_Ground_Table, plot_type="dot",max_display=21)
            for index in dependence_plot_name :
                shap.dependence_plot(index, shap_val, Back_Ground_Table)
        
        elif REGRESSOR_NAME == "RF":
            explainer = shap.TreeExplainer(self.REGRF)
            shap_val = explainer.shap_values(BackGround)
            shap.summary_plot(shap_val, Back_Ground_Table, plot_type="bar",max_display=21)
            shap.summary_plot(shap_val,Back_Ground_Table, plot_type="dot",max_display=21)
            for index in dependence_plot_name :
                shap.dependence_plot(index, shap_val, Back_Ground_Table)
        
        elif REGRESSOR_NAME == "Deep":
            explainer = shap.DeepExplainer(self.DeepREG.Regressor, BackGround)
            shap_val = explainer.shap_values(BackGround)
            shap.summary_plot(shap_val[0], Back_Ground_Table, plot_type="bar",max_display=21)
            shap.summary_plot(shap_val[0],Back_Ground_Table, plot_type="dot",max_display=21)
            for index in dependence_plot_name :
                shap.dependence_plot(index, shap_val[0], Back_Ground_Table)






    



