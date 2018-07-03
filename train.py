from sklearn import tree, svm, datasets, linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors

def nextMove(X,Y):
    x2 = np.array(list(x1), dtype=np.float)
    y2 = np.array(list(y1), dtype=np.float)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x2, y2)

    z = preprocessing.scale(data)
    z = z.reshape(1, z.shape[0])
    prediction = clf.predict(z)
    return prediction