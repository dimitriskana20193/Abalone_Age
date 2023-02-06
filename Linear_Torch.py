import numpy as np 
import torch 
import torch.nn as nn 
from sklearn.model_selection import KFold
def make_torch(X,label = False ) :
        X =  torch.from_numpy(X.astype(np.float32))
        # if the X array is the values we want to predict then we proceed with the snippet below
        if label == True:
           X = X.view(X.shape[0],1)
        return X
class LinearRegression(nn.Module): 
        def __init__(self,n_features) -> None:
            super(LinearRegression ,self).__init__()
            self.linear = nn.Linear(n_features,1)
        def forward(self,x):
            return self.linear(x)
class Log_Reg:
    def __init__(self, n_features = None, lr = 0.001, epochs = 100):
        self.n_features = n_features
        self.lr = lr
        self.epochs = epochs
    def get_params(self):
            return {'n_features': self.n_features,
            'Learning Rate': self.lr, 
            'Epochs': self.epochs}
    def set_params(self, **params): 
            for parameter, value in params.items():
                setattr(self, parameter, value)
            return self
        
    def fit(self,x,y):
        x = make_torch(x)
        y = make_torch(y,label = True)
        kf = KFold(n_splits=5, shuffle=True)
        fold_index = 0
        self.score = []
        if self.n_features == None:
                self.n_features = x.shape[1]
        model = LinearRegression(self.n_features)
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr = self.lr)
        for train_index, val_index in kf.split(x):
            fold_index += 1
            x_train, x_val = x[train_index], x[val_index]
            y_train, y_val = y[train_index], y[val_index] 
            for epoch in range(self.epochs):
                #forward pass and loss
                
                y_pred = model(x_train)
                loss = criterion(y_pred,y_train)
                #backward pass
                loss.backward()
                #update and empty gradient
                optimizer.step()
                optimizer.zero_grad()
                

                if (epoch+1) % 10 == 0:
                    print(f'epoch: {epoch +1}, loss = {loss.item():.4f}')
            self.model = model
            with torch.no_grad():
                O = model(x_val)
                los = criterion(O,y_val)
                self.score.append(los)

            return self


    def predict(self,x,y):
        x = make_torch(x)
        y = make_torch(y,label = True)
        with torch.no_grad():
            return self.model(x).round()
    def eval(self):
        return sum(self.score)/len(self.score)

    