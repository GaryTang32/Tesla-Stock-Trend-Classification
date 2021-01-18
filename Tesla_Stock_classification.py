import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import numpy as np
import xgboost as xg
from sklearn.metrics import mean_squared_error as MSE 
from matplotlib import pyplot as plt
from mlxtend.regressor import StackingCVRegressor
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
import xgboost as xg
from sklearn.feature_selection import RFECV
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
import talib

def RSI_CAL(series):
    possitive_closing = series[series>0]
    negative_closing = series[series<0]
    up = possitive_closing.mean() if len(possitive_closing) > 0 else 0
    down = abs(negative_closing.mean()) if len(negative_closing) > 0 else 0
    return 100 * up / ( up + down)
    
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
print("Program Started")
Raw_Tesla = pd.read_csv('Tesla.csv')

print(Raw_Tesla.info())

Raw_Tesla["High-Low"] = Raw_Tesla['High'] - Raw_Tesla['Low']
Raw_Tesla["Open-Close"] = Raw_Tesla['Open'] - Raw_Tesla['Close'] 
Raw_Tesla["Change"] = Raw_Tesla['Close'] - Raw_Tesla.Close.shift(1)

Raw_Tesla['RSI_10'] = Raw_Tesla.Change.rolling(10).agg(RSI_CAL)
Raw_Tesla['RSI_20'] = Raw_Tesla.Change.rolling(20).agg(RSI_CAL)
Raw_Tesla['RSI_30'] = Raw_Tesla.Change.rolling(30).agg(RSI_CAL)

Raw_Tesla['EMA_10'] = talib.EMA(Raw_Tesla.Close, timeperiod = 10)
Raw_Tesla['EMA_20'] = talib.EMA(Raw_Tesla.Close, timeperiod = 20)
Raw_Tesla['EMA_30'] = talib.EMA(Raw_Tesla.Close, timeperiod = 30)
Raw_Tesla['EMA_40'] = talib.EMA(Raw_Tesla.Close, timeperiod = 40)

Raw_Tesla["CCI_10"] = talib.CCI(Raw_Tesla.High,Raw_Tesla.Low,Raw_Tesla.Close,10)
Raw_Tesla["CCI_20"] = talib.CCI(Raw_Tesla.High,Raw_Tesla.Low,Raw_Tesla.Close,20)
Raw_Tesla["CCI_30"] = talib.CCI(Raw_Tesla.High,Raw_Tesla.Low,Raw_Tesla.Close,30)
Raw_Tesla["CCI_40"] = talib.CCI(Raw_Tesla.High,Raw_Tesla.Low,Raw_Tesla.Close,40)

Raw_Tesla['ADX_10'] = talib.ADX(Raw_Tesla.High,Raw_Tesla.Low,Raw_Tesla.Close,10)
Raw_Tesla['ADX_20'] = talib.ADX(Raw_Tesla.High,Raw_Tesla.Low,Raw_Tesla.Close,20)
Raw_Tesla['ADX_30'] = talib.ADX(Raw_Tesla.High,Raw_Tesla.Low,Raw_Tesla.Close,30)
Raw_Tesla['ADX_40'] = talib.ADX(Raw_Tesla.High,Raw_Tesla.Low,Raw_Tesla.Close,40)

Raw_Tesla['ATR_10'] = talib.ATR(Raw_Tesla.High,Raw_Tesla.Low,Raw_Tesla.Close,10)
Raw_Tesla['ATR_20'] = talib.ATR(Raw_Tesla.High,Raw_Tesla.Low,Raw_Tesla.Close,20)
Raw_Tesla['ATR_30'] = talib.ATR(Raw_Tesla.High,Raw_Tesla.Low,Raw_Tesla.Close,30)
Raw_Tesla['ATR_40'] = talib.ATR(Raw_Tesla.High,Raw_Tesla.Low,Raw_Tesla.Close,40)

Raw_Tesla.loc[Raw_Tesla.Close < Raw_Tesla.Close.shift(1),'Trend'] = 0
Raw_Tesla.loc[Raw_Tesla.Close == Raw_Tesla.Close.shift(1),'Trend'] = 1
Raw_Tesla.loc[Raw_Tesla.Close > Raw_Tesla.Close.shift(1),'Trend'] = 1

for i in range(1,11):
    text = "Trend-" + str(i) 
    Raw_Tesla[text] = Raw_Tesla.Trend.shift(i)

for i in range(5,35,5):
    text = str(i) + "-MA" 
    Raw_Tesla[text] = Raw_Tesla['Close'].rolling(window = i).mean()
    
for i in range(5,30,5):
    text = str(i) + "-STD" 
    Raw_Tesla[text] = Raw_Tesla['Close'].rolling(window = i).std()
    

Raw_Tesla['5-STD_Diff'] = Raw_Tesla['5-STD'].shift(1)
Raw_Tesla['10-STD_Diff'] = Raw_Tesla['10-STD'].shift(1)
Raw_Tesla['15-STD_Diff'] = Raw_Tesla['15-STD'].shift(1)
Raw_Tesla['20-STD_Diff'] = Raw_Tesla['20-STD'].shift(1)
Raw_Tesla['25-STD_Diff'] = Raw_Tesla['25-STD'].shift(1)


Raw_Tesla = Raw_Tesla.dropna()
Raw_Tesla = Raw_Tesla.reset_index()

for i in range(1,31):
    text = str(i) + "_day_Trend"
    Raw_Tesla[text] =  Raw_Tesla["Trend"].shift(-i)
    
for i in range(1,31):
    text = str(i) + "_day_Closing"
    Raw_Tesla[text] =  Raw_Tesla["Close"].shift(-i)
    
print(Raw_Tesla.info(verbose = True))
columns_list = list(Raw_Tesla.columns) 

for i in range(30):
    text = str(i+1) + "_day_Closing"
    columns_list.remove(text)
    
for i in range(30):
    text = str(i+1) + "_day_Trend"
    columns_list.remove(text)
    
columns_list.remove("Date")
columns_list.remove("Adj Close")
Raw_Tesla = Raw_Tesla.dropna()
Raw_Tesla = Raw_Tesla.reset_index()
Seen_Data_X = Raw_Tesla[columns_list]

Seen_Data_y = [] 
for i in range(30):
    text = str(i+1) + '_day_Closing'
    Seen_Data_y.append(Raw_Tesla[text]) 


y_Trend = pd.DataFrame()
for i in range(30):
    text = "y_Trend"+ str(i+1) 
    text1 = str(i+1) + "_day_Trend"
    y_Trend[text] = Raw_Tesla[text1]
    
print(Seen_Data_X[:10])  

GBR = GradientBoostingRegressor(verbose =0) 
LGMB = LGBMRegressor(random_state = 42, n_jobs = 4) 
KerRid = KernelRidge() 
SVR = SVR(verbose = False)
ENCV =ElasticNetCV(verbose = 0, n_jobs = 4, random_state = 42)
Lasso = LassoCV(verbose = 0, n_jobs = 4, random_state = 42)
Ridge = RidgeCV(alphas=[0.0001,0.001,0.01,0.1,1,10,100]) 
RF = RandomForestRegressor(n_jobs = 4, random_state = 42, verbose = 0)
XGBRF = xg.XGBRFRegressor()
XGB = XGBRegressor()

'''
parameters = { "cv" :[5], "normalize":[False,True],'fit_intercept':[False,True]}
SearchCV = GridSearchCV(estimator = Ridge, param_grid = parameters,verbose = 2, n_jobs =4 ,scoring = 'neg_mean_squared_error')
SearchCV.fit(train_X,train_y)
print(SearchCV.best_params_ )
Ridge = Ridge.set_params(**SearchCV.best_params_)
'''


selector = RFECV(RF, step=1, cv=10, verbose = 0, n_jobs= 4)

'''
total = 0 
for i in range(60):
    print("Predict",i+1,'Day')
    train_X, test_X, train_y, test_y = train_test_split(Seen_Data_X, Seen_Data_y[i], test_size=0.2, random_state =1, shuffle=False)
    selector = selector.fit(train_X,train_y)
    selected = selector.transform(train_X)
    print(selector.support_)
    pred = selector.predict(test_X)
    rmse = np.sqrt(MSE(test_y,pred))
    print("RMSE",rmse)
    total = total + rmse
print("Average RMSE of Random Forest for 60 days",total/60)

'''
Prediction_sum = pd.DataFrame()

Seen_Data_X_10 = Seen_Data_X[['index','Open','High','Low','Close','5-MA','10-MA']]
Seen_Data_X_20 = Seen_Data_X[['index','Open','High','Low','Close','15-MA','20-MA']]
Seen_Data_X_30 = Seen_Data_X[['index','Open','High','Low','Close','25-MA','30-MA']]

total = 0 
error_list = []

for i in range(10):
    print("Predict",i+1,'Day')
    train_X, test_X, train_y, test_y = train_test_split(Seen_Data_X_10, Seen_Data_y[i], test_size=0.2, random_state =1, shuffle=False)
    if i == 0:
        Prediction_sum['Day0'] = test_X['Close']
    RF.fit(train_X,train_y)
    pred = RF.predict(test_X)
    rmse = np.sqrt(MSE(test_y,pred))
    print("RMSE",rmse)
    total = total + rmse
    text = "Day" + str(i+1) 
    Prediction_sum[text] = pred
    error_list.append(rmse)
print("Average RMSE of Random Forest for 30 days",total/30)

for i in range(10,20):
    print("Predict",i+1,'Day')
    train_X, test_X, train_y, test_y = train_test_split(Seen_Data_X_20, Seen_Data_y[i], test_size=0.2, random_state =1, shuffle=False)
    if i == 0:
        Prediction_sum['Day0'] = test_X['Close']
    RF.fit(train_X,train_y)
    pred = RF.predict(test_X)
    rmse = np.sqrt(MSE(test_y,pred))
    print("RMSE",rmse)
    total = total + rmse
    text = "Day" + str(i+1) 
    Prediction_sum[text] = pred
    error_list.append(rmse)
print("Average RMSE of Random Forest for 30 days",total/30)

for i in range(20,30):
    print("Predict",i+1,'Day')
    train_X, test_X, train_y, test_y = train_test_split(Seen_Data_X_30, Seen_Data_y[i], test_size=0.2, random_state =1, shuffle=False)
    if i == 0:
        Prediction_sum['Day0'] = test_X['Close']
    RF.fit(train_X,train_y)
    pred = RF.predict(test_X)
    rmse = np.sqrt(MSE(test_y,pred))
    print("RMSE",rmse)
    total = total + rmse
    text = "Day" + str(i+1) 
    Prediction_sum[text] = pred
    error_list.append(rmse)
print("Average RMSE of Random Forest for 30 days",total/30)


#For ARM only
'''
for i in range(30):
    text = "trend" + str(i+1) + "Up"
    text3 = "trend" + str(i+1) + "Down"
    text1 = "Day" + str(i+1) 
    text2 = "Day" + str(i)
    Prediction_sum[text] = 0 
    Prediction_sum.loc[Prediction_sum[text1] > Prediction_sum[text2],text] = 1
    Prediction_sum[text3] = 1 
    Prediction_sum.loc[Prediction_sum[text1] > Prediction_sum[text2],text3] = 0
'''  
 
#For display only
for i in range(30):
    print('day ',i)
    text = "trend" + str(i+1) + "Up"
    text3 = "trend" + str(i+1) + "Down"
    text1 = "Day" + str(i+1) 
    text2 = "Day" + str(i)
    Prediction_sum[text] = 0
    Prediction_sum.loc[Prediction_sum[text1] > Prediction_sum[text2],text] = 1
    print(Prediction_sum[Prediction_sum[text] >= 1].shape[0])
    print(Prediction_sum[Prediction_sum[text] <= 0].shape[0])

pred = Prediction_sum[Prediction_sum.columns[-30:]]

pred.to_csv('Stock_Pattern.csv')

print(classification_report(y_Trend[-len(pred):], pred))


pred1 = pred.astype('Bool')
frequent_itemsets = apriori(pred1, min_support=0.40, use_colnames=True)
rules  = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequents_len"] = rules["consequents"].apply(lambda x: len(x))
rules['length'] = rules["antecedent_len"]+rules["consequents_len"]
rules = rules[rules['length'] > 4]
rules = rules.drop(['antecedent_len', 'consequents_len','conviction', 'length','lift','leverage'], axis=1)
temp = pd.DataFrame(rules)
print(temp) 

