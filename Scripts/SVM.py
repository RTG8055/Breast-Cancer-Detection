from sklearn import svm

def predict(train_data,test_data):
	y = [row[-1] for row in train_data]
	X = [row[:-1] for row in train_data]
	X_test = [row[:-1] for row in test_data]

	clf = svm.SVC()
	clf.fit(X,y)
	return clf.predict(X_test)
