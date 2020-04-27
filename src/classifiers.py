import argparse
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.decomposition import PCA


def main(cropped, which_model):
    train_feature = None
    train_label = None
    test_feature = None
    test_label = None
    if which_model == 'resnet':
        if cropped:
            train_feature = 'embeddings/cropped_resnet_train_feature.pkl'
            train_label = 'embeddings/cropped_resnet_train_label.pkl'
            test_feature = 'embeddings/cropped_resnet_test_feature.pkl'
            test_label = 'embeddings/cropped_resnet_test_label.pkl'
        else:
            train_feature = 'embeddings/uncropped_resnet_train_feature.pkl'
            train_label = 'embeddings/uncropped_resnet_train_label.pkl'
            test_feature = 'embeddings/uncropped_resnet_test_feature.pkl'
            test_label = 'embeddings/uncropped_resnet_test_label.pkl'
    elif which_model == 'siamese':
        if cropped:
            train_feature = 'embeddings/cropped_siamese_train_feature.pkl'
            train_label = 'embeddings/cropped_siamese_train_label.pkl'
            test_feature = 'embeddings/cropped_siamese_test_feature.pkl'
            test_label = 'embeddings/cropped_siamese_test_label.pkl'
        else:
            train_feature = 'embeddings/uncropped_siamese_train_feature.pkl'
            train_label = 'embeddings/uncropped_siamese_train_label.pkl'
            test_feature = 'embeddings/uncropped_siamese_test_feature.pkl'
            test_label = 'embeddings/uncropped_siamese_test_label.pkl'

    with open(train_feature, 'rb') as f:
        X = pickle.load(f)
    with open(train_label, 'rb') as f:
        y = pickle.load(f)
    with open(test_feature, 'rb') as f:
        Xt = pickle.load(f)
    with open(test_label, 'rb') as f:
        yt = pickle.load(f)

    # pca = PCA(n_components=200)
    # scaler = pca.fit(X)
    # X = scaler.transform(X)
    # Xt = scaler.transform(Xt)
    #
    # scaler = preprocessing.StandardScaler().fit(X)
    # X = scaler.transform(X)
    # Xt = scaler.transform(Xt)

    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X, y)
    pred = neigh.predict(Xt)
    print("1-NN: Accuracy:{}, F1:{}".format(accuracy_score(yt, pred), f1_score(yt, pred, average='macro')))

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y)
    pred = neigh.predict(Xt)
    print("3-NN: Accuracy:{}, F1:{}".format(accuracy_score(yt, pred), f1_score(yt, pred, average='macro')))

    clf = svm.SVC(max_iter=10000, decision_function_shape='ovo')
    clf.fit(X, y)
    pred = clf.predict(Xt)
    print("SVM(One-Against-One): Accuracy:{}, F1:{}".format(accuracy_score(yt, pred), f1_score(yt, pred, average='macro')))

    clf = LogisticRegression(random_state=0, max_iter=10000).fit(X, y)
    pred = clf.predict(Xt)
    print("Logistic Regression: Accuracy:{}, F1:{}".format(accuracy_score(yt, pred), f1_score(yt, pred, average='macro')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cropped", action='store_true', help="whether to use cropped dataset")
    parser.add_argument("--model", type=str, choices=['resnet', 'siamese'], default='resnet',
                        help="which trained model to use")
    args = parser.parse_args()

    main(args.cropped, args.model)
