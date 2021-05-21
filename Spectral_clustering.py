
# 谱聚类



import numpy as np
from numpy.core.fromnumeric import diagonal




# def calEuclidDistanceMatrix(X):
#     X = np.array(X)
#     S = np.zeros((len(X), len(X)))
#     for i in range(len(X)):
#         for j in range(i+1, len(X)):
#             S[i][j] = 1.0 * euclidDistance(X[i], X[j])
#             S[j][i] = S[i][j]
#     return S






class spectral():
    def __init__(self,X,k) -> None:
        self.X = X
        self.k = k

    def euclidDistance(self,x1, x2, use_l21=False):
        res = np.sum((x1-x2)**2)
        if use_l21:
            res = np.sqrt(res)
        return res

    def calAdjacencyMatrix_RBF(self,sigma=1/3):
        N = len(self.X)
        W = np.zeros((N,N))

        for i in range(N) :
            for j in range(i+1, len(X)):
                W[i][j] = np.exp(self.euclidDistance(X[i], X[j])*(-1/2)*(sigma**2))       #[sigma] 越大表示样本点与距离较远的样本点的相似度越大
                W[j][i] = W[i][j]
        
        return W

    def calLaplacianMatrix(self,W):
        # 计算度矩阵
        D1 = np.sum(W,axis=1)  # 按第1个维度（从0开始）方向求和
        D = np.diag(D1)
        # 计算标准化拉普拉斯矩阵
        L = D -W
        sqrtDegreeMatrix = np.diag(1.0 / (D1 ** (0.5)))
        # 标准化
        L  = np.dot(np.dot(sqrtDegreeMatrix,L),sqrtDegreeMatrix)
        # print(L)
        return L

    def run(self):
        W = self.calAdjacencyMatrix_RBF()
        L = self.calLaplacianMatrix(W)
        # 求特征值和特征向量
        lam,V = np.linalg.eig(L)
        # 重新排列特征值和特征向量
        index = np.argsort(lam)     #获得重排索引
        lam = lam[index]
        V = V[:,index]
        
        F = V[:,np.arange(self.k)]       # top k个特征向量
        for i in np.arange(len(F)):
            F[i] =F[i] / np.linalg.norm(F[i])    # F[i]和F[j]相似就会聚为一类
        


if __name__ == '__main__':
    X = np.array([[1,2,3,4],[2,4,5,9],[2,3,1,5]])
    k =3
    spec = spectral(X,k)
    spec.run()
    # W = calAdjacencyMatrix_RBF(X)
    # # 计算标准阿化的拉普拉斯矩阵
    # L = calLaplacianMatrix(W)
    # # 求特征值和特征向量
    # lam,V = np.linalg.eig(L)
    # # 重新排列特征值和特征向量
    # index = np.argsort(lam)     #获得重排索引
    # lam = lam[index]
    # V = V[:,index]




    