
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def gmm(X, n_components):
    X = StandardScaler().fit_transform(X)
    gmm = GaussianMixture(n_components, random_state=42)
    probs = gmm.fit(X).predict_proba(X)
    return probs

def pca_select_important_features(data, n_components, top_n_features=5):
    """
    对数据进行PCA降维并选出贡献最大的前n个原始特征
    
    参数：
        data: pandas DataFrame，原始特征数据（不包含标签）
        n_components: int，保留的主成分数量
        top_n_features: int，从PCA中找出前n个最重要的原始特征

    返回：
        important_features: list，重要性前n个原始特征名称
        pca_data: DataFrame，经过PCA降维后的数据
    """

    # 标准化数据（非常重要）
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # 应用PCA
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)

    # 获取PCA分量对原始特征的系数（loadings）
    loadings = pca.components_

    # 计算每个原始特征对所有主成分的总重要性（取绝对值求和）
    feature_importance = np.sum(np.abs(loadings), axis=0)

    # 创建特征重要性Series，并按值排序
    feature_names = data.columns
    importance_series = pd.Series(feature_importance, index=feature_names)
    important_features = importance_series.sort_values(ascending=False).head(top_n_features).index.tolist()

    # 将PCA降维结果转换为DataFrame（可选）
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(pca_data, columns=pca_columns)

    return important_features, pca_df


def fuzzy_cluster_pca_data(X, n_clusters=3, m=2.0, error=0.005, maxiter=1000):
    """
    对 PCA 降维后的数据进行模糊聚类（Fuzzy C-Means）

    参数:
        X : ndarray or DataFrame, shape = (n_samples, n_features)
            PCA 降维后的输入数据
        n_clusters : int
            聚类数量
        m : float
            模糊指数，一般为 2（越大越模糊）
        error : float
            收敛容差
        maxiter : int
            最大迭代次数

    返回:
        labels : ndarray, shape = (n_samples,)
            每个样本最可能的聚类标签
        u : ndarray, shape = (n_clusters, n_samples)
            隶属度矩阵，每个样本属于每个簇的程度
        centers : ndarray, shape = (n_clusters, n_features)
            聚类中心坐标
    """
    if isinstance(X, pd.DataFrame):
        data = X.T.values  # shape: (features, samples)
    else:
        data = X.T  # numpy 的情况

    # 模糊 C 均值聚类
    centers, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data=data,
        c=n_clusters,
        m=m,
        error=error,
        maxiter=maxiter,
        init=None
    )

    # 获取每个样本所属的类别标签（隶属度最大的位置）
    labels = np.argmax(u, axis=0)

    return labels, u, centers