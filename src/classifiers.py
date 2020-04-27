import argparse
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.decomposition import PCA


def main(cropped):
    if cropped:
        with open('embeddings/cropped_siamese_train_feature.pkl', 'rb') as f:
            X = pickle.load(f)
        with open('embeddings/cropped_siamese_train_label.pkl', 'rb') as f:
            y = pickle.load(f)
        with open('embeddings/cropped_siamese_test_feature.pkl', 'rb') as f:
            Xt = pickle.load(f)
        with open('embeddings/cropped_siamese_test_label.pkl', 'rb') as f:
            yt = pickle.load(f)
    else:
        with open('embeddings/uncropped_siamese_train_feature.pkl', 'rb') as f:
            X = pickle.load(f)
        with open('embeddings/uncropped_siamese_train_label.pkl', 'rb') as f:
            y = pickle.load(f)
        with open('embeddings/uncropped_siamese_test_feature.pkl', 'rb') as f:
            Xt = pickle.load(f)
        with open('embeddings/uncropped_siamese_test_label.pkl', 'rb') as f:
            yt = pickle.load(f)

    pca = PCA(n_components=50)
    scaler = pca.fit(X)
    X = scaler.transform(X)
    Xt = scaler.transform(Xt)

    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    Xt = scaler.transform(Xt)

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
    parser.add_argument("--cropped", action='store_true', help="whether to train on cropped dataset")
    args = parser.parse_args()
    main(args.cropped)
