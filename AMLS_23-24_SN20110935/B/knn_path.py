#Reference: https://medium.com/@agrawalsam1997/hyperparameter-tuning-of-knn-classifier-a32f31af25c7

import os
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

def knn_path():
    dir = os.path.dirname(__file__).split(os.sep)
    dir[len(dir)-1] = 'Datasets'
    dir.append('PathMNIST')
    dir.append('pathmnist.npz')
    data_file = os.sep.join(dir)
    npz_file = np.load(data_file)
    lst = npz_file.files
    for item in lst:    
        if (item == 'train_images'):
            X_train_images = npz_file[item]
            X_train = X_train_images.reshape(X_train_images.shape[0], X_train_images.shape[1]*X_train_images.shape[2]*X_train_images.shape[3])
        if (item == 'train_labels'):
            y_train = npz_file[item].flatten()
        if (item == 'val_images'):
            X_val_images = npz_file[item]
            X_val = X_val_images.reshape(X_val_images.shape[0], X_val_images.shape[1]*X_val_images.shape[2]*X_train_images.shape[3])
        if (item == 'val_labels'):
            y_val = npz_file[item].flatten()
        if (item == 'test_images'):
            X_test_images = npz_file[item]
            X_test = X_test_images.reshape(X_test_images.shape[0], X_test_images.shape[1]*X_test_images.shape[2]*X_train_images.shape[3])
        if (item == 'test_labels'):
            y_test = npz_file[item].flatten()

    val_score = {}
    val_predict = {}
    test_predict = {}
    k_neighbors = range(1,17,2)
    #k_neighbors = [100, 200, 300, 500]

    print('knn hyperparameter tuning using for loop')
    start_time = time.time()
    for k in k_neighbors: #hyperparameter tuning
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        val_score[k] = knn.score(X_val, y_val)
        print('k=%d, val acc:%.6f' % (k, val_score[k]))
    end_time = time.time()
    print('Time spent for train is %.3f seconds.' % (end_time - start_time))

    for key, value in val_score.items():
        if value==max(val_score.values()):
            best_k = key
    print(f'Hyperparameter tuning using for loop: the best k from evaluation is {best_k}.')

    print('Test in progress using the best k.')
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    best_test_acc = knn.score(X_test, y_test)
    # for i in range(0, len(y_test)):
    #     if test_predict[k][i] != y_test[i]:
    #         print(f'Predicted label for test data no. {i+1} is {test_predict[k][i]} but actual label is {y_test[i]}')
    print('Hyperparameter tuning using for loop: the best test acc:%.3f.' % (best_test_acc))

    print('KNN hyperparameter tuning using k-fold cross validation.')
    start_time = time.time()
    kf=KFold(n_splits=5, shuffle=True, random_state=42)
    parameter={'n_neighbors': np.arange(1, 17, 2)}
    #parameter={'n_neighbors': [100, 200, 300, 500]}
    knn=KNeighborsClassifier()
    knn_cv=GridSearchCV(knn, param_grid=parameter, cv=kf, verbose=1)
    #Merge the train/val split for k-fold cross validation.
    knn_cv.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0))
    best_k = knn_cv.best_params_["n_neighbors"]
    print(f'Hyperparameter tuning using k-fold cross validation: the best k={best_k}.')
    end_time = time.time()
    print('Time spent for cross validation is %.3f seconds.' % (end_time - start_time))

    print('Test in progress using the best k.')
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    best_test_acc = knn.score(X_test, y_test)
    print('Hyperparameter tuning using for loop: the best test acc:%.3f.' % (best_test_acc))
