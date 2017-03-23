from sklearn.neighbors import KNeighborsClassifier

from src.data import Data

data = Data("img")
data.load_images()
#data.load_pickle("images.pickle")
#X_train, X_test, y_train, y_test = data.train_test_split()
#print(X_test[0].shape, y_test)

neigh = KNeighborsClassifier(n_neighbors=1)
#neigh.fit(X_train,  y_train)

#print(neigh.predict(X_test[0]),y_test[0])
