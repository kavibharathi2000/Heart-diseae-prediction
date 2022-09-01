
"""
Heart Disease Prediction

"""

# importion
import pandas as  pd
import numpy as np


# main
if __name__ == "__main__":

    # loading the dataset
    file = pd.read_csv("dataset/heart.csv")
    x_data = file.iloc[:,0:-1].values 
    y_data = file.iloc[:,-1:].values 

    # data preprocessing
    from sklearn.preprocessing import StandardScaler
    der = StandardScaler()
    x_data = der.fit_transform(x_data)


    # train and test split of dataset
    from sklearn.model_selection import train_test_split
    x_train , x_test, y_train , y_test = train_test_split(x_data, y_data, test_size= 0.25) 

    # fitting the model
    from sklearn.linear_model import LogisticRegression
    reg = LogisticRegression()
    reg.fit(x_train, np.ravel(y_train))

    #predicting the model;
    output = reg.predict(x_test)

    # accuracy score
    from sklearn.metrics import accuracy_score
    ac = accuracy_score(y_test, output)
    print("The Accuracy Score is : ", ac)


