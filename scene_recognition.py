#By Avneet Singh Saluja (5586107)
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))
    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(
            PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


def compute_dsift(img, stride, size=16):
    # To do
    r = size//2
    sift = cv2.xfeatures2d.SIFT_create()
    kp = [cv2.KeyPoint(x, y, r) for y in range(r, img.shape[0]-r, stride)
          for x in range(r, img.shape[1]-r, stride)]
    kps, dense_feature = sift.compute(img, kp)
    return dense_feature


def get_tiny_image(img, output_size):
    # To do
    tiny_img = cv2.resize(img, output_size)
    feature = tiny_img.flatten()
    feature = feature - np.mean(feature)
    feature = feature / np.linalg.norm(feature)
    return feature

def findMajorityLabel(indxs, labels):
    mapLabelCount = {}
    for indx in indxs:
        label = labels[indx]
        if label in mapLabelCount.keys():
            mapLabelCount[label] = mapLabelCount[label] + 1
        else:
            mapLabelCount[label] = 1
    return sorted(mapLabelCount, key=mapLabelCount.get, reverse=True)[0]

def predict_knn(feature_train, label_train, feature_test, k):
    # To do
    label_test_pred = np.asarray([])
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(feature_train)
    for ft in feature_test:
        dist, k_neigh_ids = neigh.kneighbors([ft])
        majority_label = findMajorityLabel(k_neigh_ids[0], label_train)
        label_test_pred = np.append(label_test_pred, majority_label)
    return label_test_pred

def calcConfusionAndAcc(pred_labels, true_labels, label_classes):
    N = len(label_classes)
    confusion = np.zeros(shape = [N, N])
    mapLabelToIndx = {}
    for i, label in enumerate(label_classes):
        mapLabelToIndx[label] = i

    for i, pred_label in enumerate(pred_labels):
        true_label = true_labels[i]
        confusion[mapLabelToIndx[true_label]][mapLabelToIndx[pred_label]]+=1

    accuracy = 0
    for i in range(N):
        accuracy += confusion[i][i]
    accuracy/=np.sum(confusion)

    return confusion, accuracy

def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    feature_train = np.empty(shape=[0, 256])
    feature_test = np.empty(shape=[0, 256])
    k = 7
    for img_train_path in img_train_list:
        img = cv2.imread(img_train_path, 0)
        tiny_img_feature = get_tiny_image(img, (16, 16))
        feature_train = np.vstack((feature_train, tiny_img_feature))

    for img_test_path in img_test_list:
        img = cv2.imread(img_test_path, 0)
        tiny_img_feature = get_tiny_image(img, (16, 16))
        feature_test = np.vstack((feature_test, tiny_img_feature))

    pred_labels = predict_knn(feature_train, label_train_list, feature_test, k)
    confusion, accuracy = calcConfusionAndAcc(pred_labels, label_test_list, label_classes)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def build_visual_dictionary(dense_feature_list, dic_size):
    # To do
    kmeans = KMeans(n_clusters=dic_size, random_state=0, n_init=15, max_iter=500).fit(dense_feature_list)
    vocab = kmeans.cluster_centers_
    return vocab


def compute_bow(feature, vocab):
    # To do
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(vocab)
    bow_feature = np.zeros(vocab.shape[0])
    for ft in feature:
        dist, id = neigh.kneighbors([ft])
        bow_feature[id[0]] += 1
    bow_feature = bow_feature / np.linalg.norm(bow_feature)
    return bow_feature


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    k = 8
    bow_size = 100
    patch_size = 64
    stride = 64
    dense_feature_list = np.empty(shape=[0, 128])
    for img_train_path in img_train_list:
        img = cv2.imread(img_train_path, 0)
        dense_feature = compute_dsift(img, stride, patch_size)
        for d_feature in dense_feature:
            dense_feature_list = np.vstack((dense_feature_list, d_feature.reshape(1, -1)))

    vocab = build_visual_dictionary(dense_feature_list, bow_size)
    #np.save('vocab2.npy', vocab)
    #vocab = np.load('vocab2.npy')

    feature_train = np.empty(shape=[0, bow_size])
    feature_test = np.empty(shape=[0, bow_size])

    for img_train_path in img_train_list:
        img = cv2.imread(img_train_path, 0)
        dense_feature = compute_dsift(img, stride, patch_size)
        bow_feature = compute_bow(dense_feature, vocab)
        feature_train = np.vstack((feature_train, bow_feature))

    for img_test_path in img_test_list:
        img = cv2.imread(img_test_path, 0)
        dense_feature = compute_dsift(img, stride, patch_size)
        bow_feature = compute_bow(dense_feature, vocab)
        feature_test = np.vstack((feature_test, bow_feature))

    pred_labels = predict_knn(feature_train, label_train_list, feature_test, k)
    confusion, accuracy = calcConfusionAndAcc(pred_labels, label_test_list, label_classes)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test, n_classes):
    # To do
    linear_svm_models = []
    for i in range(n_classes):
        labels = label_train == i
        labels = labels.astype(int)
        clf = LinearSVC(C=0.25)
        clf.fit(feature_train, labels)
        linear_svm_models.append(clf)

    label_test_pred = []
    for ft in feature_test:
        label_test_pred.append(np.argmax(np.asarray([clf.decision_function([ft])[0] for clf in linear_svm_models])))

    return np.asarray(label_test_pred)


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do

    bow_size = 500
    patch_size = 50
    stride = 50
    dense_feature_list = np.empty(shape=[0, 128])
    for img_train_path in img_train_list:
        img = cv2.imread(img_train_path, 0)
        dense_feature = compute_dsift(img, stride, patch_size)
        for d_feature in dense_feature:
            dense_feature_list = np.vstack((dense_feature_list, d_feature.reshape(1, -1)))

    vocab = build_visual_dictionary(dense_feature_list, bow_size)
    #np.save('vocab_svm.npy', vocab)
    #vocab = np.load('vocab_svm.npy')
    feature_train = np.empty(shape=[0, bow_size])
    feature_test = np.empty(shape=[0, bow_size])

    for img_train_path in img_train_list:
        img = cv2.imread(img_train_path, 0)
        dense_feature = compute_dsift(img, stride, patch_size)
        bow_feature = compute_bow(dense_feature, vocab)

        feature_train = np.vstack((feature_train, bow_feature))

    for img_test_path in img_test_list:
        img = cv2.imread(img_test_path, 0)
        dense_feature = compute_dsift(img, stride, patch_size)
        bow_feature = compute_bow(dense_feature, vocab)
        feature_test = np.vstack((feature_test, bow_feature))

    label_train = np.asarray([])
    for lt in label_train_list:
        label_train = np.append(label_train, label_classes.index(lt))

    pred_labels = predict_svm(feature_train, label_train, feature_test, 15)
    pred_labels = np.asarray([label_classes[pl] for pl in pred_labels])
    confusion, accuracy = calcConfusionAndAcc(pred_labels, label_test_list, label_classes)
    visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # To do: replace with your dataset path

    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info(
        "./scene_classification_data")

    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
