import pandas as pd 
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

Data = pd.read_csv('/home/dimitriskana/workspace/Project_Aging_meat/abalone.data.csv', names=['Sex', 'Length', 'Diameter',
'Height', 'Whole Weight', 'Shucked Weight', 'Viscera Weight','Shell Weight', 'Rings'])

#Data['Volume'] = Data['Height']*Data['Diameter']*Data['Length']


#pd.plotting.scatter_matrix(Data,figsize = (12,8))
# we need to one-hot encode the categorical data
one_hot = pd.get_dummies(Data['Sex'])
#Data = Data.drop(['Sex','Length', 'Diameter',
#'Height'], axis =1).join(one_hot)
Data = Data.drop(['Sex'], axis =1).join(one_hot)
X = Data.copy().drop('Rings', axis = 1).to_numpy()
y = Data['Rings'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, 
test_size=0.33, random_state=1234)


from Linear_Torch import Log_Reg as LR
# lr starts at 0.001 by default and epochs at 100
d = LR( lr = 0.0001,epochs = 20000 )
d.fit(X_train,y_train)
y_pred = d.predict(X_test, y_test).numpy()
acc = accuracy_score(y_test, y_pred)
print(f'The accuracy of the model is : {acc :.4f}')
print(f'The mean value of the score cross-validated data is: {d.eval() :.4f}')
a = 5