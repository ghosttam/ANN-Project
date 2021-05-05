import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# import data
col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
             'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(r'diabetes.csv', header=0, names=col_names)
print(data)

# feature selection
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[feature_cols]
X = data.drop("Outcome", axis=1)
# Target variable
y = data.Outcome

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print("Training Data :", X_train.shape)
print("Testing Data : ", X_test.shape)




#mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
#mlp.fit(X_train, y_train.values.ravel())

#predictions = mlp.predict(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train,y_train)

predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

print("Training Prediction :")
print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))

print("Testing Prediction :")
print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))

categorical_data= data
encoder= LabelEncoder()
for i in categorical_data:
  data[i] = encoder.fit_transform(data[i])
  data.dtypes

X1=data[feature_cols]
y1=data.Outcome
X_train1,X_test1,y_train1,y_test1=train_test_split(X1,y1,test_size=0.3,random_state=0)
pred_prob=mlp.predict_proba(X_test1)

fpr, tpr, threshold=roc_curve(y_test1,pred_prob[:,1], pos_label=1)
auc_NNmodel=auc(fpr,tpr)
plt.title('ROC & AUC Curve For Neural Network Model')
plt.plot(fpr, tpr, 'r--',label = 'ROC' )
plt.plot(fpr, tpr, marker='.', label = 'AUC= %0.2f' % auc_NNmodel)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
