from sklearn.metrics import f1_score, roc_curve, auc

def simulate(clf, X_train, X_test, y_train, y_test, roc = False):
    # fit using training data
    clf.fit(X_train, y_train)
    # calculate predictions
    y_hat_train = clf.predict(X_train)
    y_hat_test = clf.predict(X_test)
    # STORE TRAINING AND TEST SCORES
    acc_train = round(clf.score(X_train, y_train), 3)
    acc_test = round(clf.score(X_test, y_test), 3)
    
    f1_train = round(f1_score(y_train, y_hat_train), 3)
    f1_test = round(f1_score(y_test, y_hat_test), 3)
    # only will record test AUC
    fpr, tpr, thresh = roc_curve(y_test, y_hat_test)
    test_auc = round(auc(fpr, tpr), 3)

    stats = [acc_train, acc_test, f1_train, f1_test, test_auc]

    if roc:
    	return stats, [fpr, tpr]
    else:
    	return stats