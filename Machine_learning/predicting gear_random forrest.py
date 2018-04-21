import pandas as pd
import numpy as np
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# import statsmodels.formula.api as smapi
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

base_path="Users\mayank\Documents\mytensorflow\Datasets\"
file_name="MY_refine_engine_data.xlsx"
data=pd.read_excel(base_path+file_name)
'''
#ploting wheel rpm and gear
plt.scatter(data['WhlRPM_FL[rpm]']/10,data['Gr[]'],color='b')
# plt.plot(x_train,regresion_data.predict(x_train),color='r')
plt.xlabel("Wheel_RPM")
plt.ylabel("Gear")
plt.show()

#ploting wheel rpm and gear
plt.scatter(data['EngRPM[rpm]'],data['Gr[]'],color='r')
# plt.plot(x_train,regresion_data.predict(x_train),color='r')
plt.xlabel("Engine_RPM")
plt.ylabel("Gear")
plt.show()
#ploting wheel accped and gear
plt.scatter(data['AccelPdlPosn[%]'],data['Gr[]'],color='b')
# plt.plot(x_train,regresion_data.predict(x_train),color='r')
plt.xlabel("Accped")
plt.ylabel("Gear")
plt.show()
#ploting wheel engine torque and gear
plt.scatter(data['EngTrq[Nm]'],data['Gr[]'],color='g')
# plt.plot(x_train,regresion_data.predict(x_train),color='r')
plt.xlabel("Engine Torque")
plt.ylabel("Gear")
plt.show()'''

features=data.iloc[:,:-1].values
labels=data.iloc[:,-1].values
print (features)
# print labels

x_train,xtest,y_train,ytest=train_test_split(features,labels,test_size=.25,random_state=2355)
print (x_train.shape)
myclass=RandomForestClassifier(n_estimators=100, oob_score=True,random_state=2355)
myclass.fit(x_train,y_train)

from sklearn.metrics import accuracy_score
predicted = myclass.predict(xtest)
print ("predicted Gear :", predicted)
print
print("Real Gear are", ytest)
accuracy = accuracy_score(ytest, predicted)
print
print ("the Accuracy of Model is : ",accuracy)

# #ploting wheel engine torque and gear
# plt.scatter(ytest,predicted,color='g')
# # plt.plot(x_train,regresion_data.predict(x_train),color='r')
# plt.xlabel("Predicted Gear")
# plt.ylabel("Real Gear")
# plt.show()


# print myclass.max_leaf_nodes
df=pd.DataFrame()
df['Real_gear']= ytest
df['predicted gear']= predicted
dat12=['1','2','3','4','5','6','7','8','13']
labedled = [dat12,dat12]
# print confusion_matrix(ytest, predicted, labels=labedled)
# writer = pd.ExcelWriter('finalprediction1.xlsx', engine='xlsxwriter')
#
# # Convert the dataframe to an XlsxWriter Excel object.
# df.to_excel(writer, sheet_name='Sheet1')
#
# # Close the Pandas Excel writer and output the Excel file.
# writer.save()
