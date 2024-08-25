from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
# from sample_weight_optimization_margin import get_optimized_weights, get_data_k_near_margins
import numpy as np
import heapq


# perform data normalization of condition features before feature selection
def data_preprocess_normalize(X, type='min_max'):
    assert X.shape[0] > 0

    if type == 'mean_std':
        data_scaler = StandardScaler()  # record mean and std for test set
    elif type == 'min_max':
        data_scaler = MinMaxScaler()
    else:
        return X

    X_norm = data_scaler.fit_transform(X)
    return X_norm, data_scaler


def flip_sigmoid_function(x, beta=1.0):
    # flipped sigmoid activation function
    return 1 / (1 + np.exp(beta * x))


def derivative_function(x, beta=1.0):
    # derivative of flipped sigmoid activation function
    return - (beta * np.exp(beta * x)) / ((1 + np.exp(beta * x)) ** 2)


def get_minus_margin(k_miss_dist, k_hits_dist):
    return np.array(k_miss_dist) - np.array(k_hits_dist)


# get k nearest hits and misses of sample x_i given weights
def get_k_nearest_miss_hit(X, y, x_i, k=3, sample_weights=None, feas_weights=None):
    # initialize sample weights
    if sample_weights is None:
        sample_weights = np.array([1.0] * len(y))

    # compute the distance vector between the sample xi to all samples in X, including xi itself
    metric = 'euclidean'  # ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
    vec_dists = pairwise_distances(X, [X[x_i, :]], metric=metric)  # X and Y must be an array
    vec_dists = vec_dists.flatten() * sample_weights

    # k nearest hits
    cls_idx = np.where(y == y[x_i])[0]       # the idx of samples in the same class
    vec_hits = vec_dists[cls_idx]            # the distances between samples xi to the samples in the same class

    # the smallest k values and indexes
    smallest_idx = heapq.nsmallest(k+1, range(len(vec_hits)), vec_hits.__getitem__)
    smallest_val = heapq.nsmallest(k+1, vec_hits)
    # remove the sample x itself
    smallest_idx = smallest_idx[1:]
    smallest_val = smallest_val[1:]

    k_hits_dist = smallest_val               # k nearest hit distances to xi
    k_hits_idx = cls_idx[smallest_idx]       # k nearest hit samples to xi

    # k nearest misses
    cls_idx = np.where(y != y[x_i])[0]       # the idx of samples in the different classes
    vec_miss = vec_dists[cls_idx]            # the distances between samples xi to the samples in the different classes

    # the smallest k values and indexes
    smallest_idx = heapq.nsmallest(k, range(len(vec_miss)), vec_miss.__getitem__)
    smallest_val = heapq.nsmallest(k, vec_miss)

    k_miss_dist = smallest_val               # k nearest miss distances to xi
    k_miss_idx = cls_idx[smallest_idx]       # k nearest miss samples to xi

    weighted_margins = get_minus_margin(k_miss_dist, k_hits_dist)

    return [weighted_margins, k_miss_dist, k_miss_idx, k_hits_dist, k_hits_idx]


# get the sum of the k-order weighted margin of all samples
def get_data_k_near_margins(X, y, k=1, sample_weights=None, feas_weights=None):

    loss_sum, margin_sum, margin_all, miss_dist_all, miss_all, hits_dist_all, hits_all = 0, 0, [], [], [], [], []
    for i in range(len(y)):
        # [weight_margins, k_miss_dist, k_miss_idx, k_hits_dist, k_hits_idx]
        xi_miss_hits = get_k_nearest_miss_hit(X, y, i, k, sample_weights, feas_weights)
        margin_all.append(xi_miss_hits[0])
        miss_dist_all.append(xi_miss_hits[1])
        miss_all.append(xi_miss_hits[2])
        hits_dist_all.append(xi_miss_hits[3])
        hits_all.append(xi_miss_hits[4])
        margin_sum += np.mean(xi_miss_hits[0])

    # get the mean and standard deviation of margins of samples in each class
    # get all possible decision values
    all_des_val = np.unique(y)
    vec_margin = np.mean(np.array(margin_all), axis=1)

    cls_margins = np.empty([len(all_des_val), 2], dtype='float')
    for i in range(len(all_des_val)):
        cls_idx = np.where(y == all_des_val[i])[0]  # the idx of samples in the same class
        cls_margin = vec_margin[cls_idx]
        cls_margins[i][0] = np.mean(cls_margin)
        cls_margins[i][1] = np.std(cls_margin)

    # get overall loss
    for i in range(len(y)):
        xi_margin = vec_margin[i]
        cls_idx = np.where(all_des_val == y[i])[0]
        xi_margin = xi_margin - cls_margins[cls_idx[0]][0]  # margin minus class margin mean for zero-centered
        loss_sum += flip_sigmoid_function(xi_margin)

    # loss_avg = loss_sum / len(y)
    loss = loss_sum
    # loss = - margin_sum

    # total weighted margins, all weighted margins, k nearest miss of all samples, k nearest hit of all samples
    return [loss, margin_sum, cls_margins, np.array(margin_all), np.array(miss_dist_all), np.array(miss_all),
            np.array(hits_dist_all), np.array(hits_all)]


# k-order average granule margin given a set of features
def agm(X, y, k=3, p=None):
    (n_rows, _) = X.shape
    assert n_rows == len(y)

    # compute the k nearest miss and hits of all samples
    # [loss, margin_sum, cls_margins, margin_all, miss_dist_all, miss_all, hits_dist_all, hits_all]
    _, margins_sum, cls_margins, _, _, _, _, _ = get_data_k_near_margins(X, y, k, p)

    return margins_sum


# @ perform average granule margin-based feature selection
def agm_fs(X, y, k=3, p=None, epsilon=0.0001):
    """
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be discrete or continuous feature
    y: {numpy array}, shape (n_samples,)
        input class labels, guaranteed to be discrete
    k: {int}
        the order of neighborhood margin, default value k=3
    epsilon: {float}
        stop criteria - if the improvement is less or equal than epsilon, the algorithm stops.

    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature if core feature is not added in advance
    sig_val: {numpy array}, shape: (n_features,)
        corresponding significance value of selected features

    """
    if p is None:
        p = np.array([1.0] * len(y))

    (n_rows, m_cols) = X.shape

    # ************02: perform feature selection************
    all_mes = agm(X, y, k, p)  # overall margin under all condition features

    idx, cur_mes, pre_mes = 0, 0, -1000
    F = []         # index of selected features
    feas_mes = []  # margin between iteratively selected features and decision class: relevance
    feas_rel = []  # the correlation between each feature and decision

    feas_mes.append(all_mes)

    # iteratively select features
    while True:
        # compute agm for each candidate attribute
        fea_tmp = np.array([-1.0] * m_cols)
        for i in range(m_cols):
            F_temp = F.copy()
            if i not in F:
                F_temp.append(i)
                fea_tmp[i] = agm(X[:, F_temp], y, k, p)

        # return agm of each feature w.r.t decision
        if len(F) == 0:
            feas_rel = fea_tmp.copy()

        # select the feature with highest significance
        idx = np.argmax(fea_tmp)  # index of the feature with highest significance
        cur_mes = fea_tmp[idx]

        if cur_mes < pre_mes or m_cols == len(F):
            break

        F.append(idx)
        feas_mes.append(cur_mes)
        pre_mes = cur_mes

    return F, feas_mes, feas_rel


def test_FS():
    data_name = 'dow-jones-index.data'
    file_path = './data/' + data_name

    # **********01: load data**********
    data = np.loadtxt(file_path, dtype=float, delimiter=',')
    (_, m_features) = data.shape

    m_features = m_features - 1
    X = data[:, 0:m_features]
    y = data[:, -1].astype('int8')
    print("01:loading data: ", data.shape)

    # preprocess data
    X_transform, _ = data_preprocess_normalize(X, 'min_max')

    # weight learning
    vec_weights = np.array([1] * len(y))
    # ********** to be released **********
    """
    k, l_rate, init_weights = 3, 0.001, np.array([0.5] * len(y))
    vec_weights, all_loss = get_optimized_weights(X_transform, y, k, l_rate, init_weights)
    print(vec_weights)
    """

    feas_sel_agm, _, _ = agm_fs(X_transform, y, 3, vec_weights, 0)
    print(feas_sel_agm, len(feas_sel_agm))

    # **********03: cross-validation for performance evaluation**********
    clf_knn = KNeighborsClassifier(n_neighbors=3)
    clf_svm = SVC(kernel='rbf')
    kf = KFold(n_splits=10, shuffle=True, random_state=0)

    reduced_X = X_transform[:, sorted(feas_sel_agm)]

    scores = cross_val_score(clf_knn, reduced_X, y, cv=kf)
    print("\n02:Performance with agm KNN: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print(scores)

    scores = cross_val_score(clf_knn, X_transform, y, cv=kf)
    print("\n02:Performance with raw features KNN: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print(scores)

    scores = cross_val_score(clf_svm, reduced_X, y, cv=kf)
    print("\n02:Performance with agm SVM: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print(scores)

    scores = cross_val_score(clf_svm, X_transform, y, cv=kf)
    print("\n02:Performance with raw features SVM: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
    print(scores)


if __name__ == '__main__':
    # test average granule margin-based feature selection
    test_FS()
