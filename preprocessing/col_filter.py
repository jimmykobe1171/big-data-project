
import pickle
import csv

from scipy.sparse import *
import numpy as np
import random

data_dir = './5000/'

def MSE(A, U_T, V_T, Omega):
    mse = 0
    num = len(Omega)

    Ahat = np.dot(U_T, V_T.T)

    for (i, j) in Omega:
        mse += abs(Ahat[i, j] - A[i, j])

    return mse / num


def distributed_MSE(A, U_T, V_T, Omega):
    """
    :param:
        A: csr_matrix
        U_T: csr_matrix
        V_T: csr_matrix
    """

    mse = 0
    num = len(Omega)

    for (i, j) in Omega:
        ahat_ij = np.dot(U_T[i], V_T[j].T)[0, 0]

        mse += abs(ahat_ij - A[i, j])

    return mse / num

def SGD(A, U_T, V_T, Omega, U_T_Prime, V_T_Prime):
    """
    :param:
        A: np.array/csr_matrix
        U_T: np.array/csr_matrix
        V_T: np.array/csr_matrix
    """
    step_size = 0.001

    mse_list = []

    for _ in range(100):
        # One pass
        random.shuffle(Omega)

        for (i, j) in Omega:
            tmp = np.dot(U_T[i], V_T[j].T) - A[i, j]

            # U_T[i] = U_T[i] - step_size * (2 * tmp * V_T[j] - 1 * V_T_Prime[j])
            # V_T[j] = V_T[j] - step_size * (2 * tmp * U_T[i] - 1 * U_T_Prime[i])

            U_T[i] = U_T[i] - step_size * 2 * tmp * V_T[j]
            V_T[j] = V_T[j] - step_size * 2 * tmp * U_T[i]

        mse = MSE(A, U_T, V_T, Omega)
        mse_list.append(mse)

    for mse in mse_list:
        print(mse)



def build_sparse_matrix(m, n, sparse=True):
    ratings_csv = data_dir + 'ratings.csv'

    Omega = []
    u_list = []
    r_list = []
    ratings_list = []

    f = open(ratings_csv, 'r')
    lines = csv.reader(f)

    for line_list in lines:
        uid = int(line_list[0])
        rid = int(line_list[1])

        ratings = float(line_list[2])

        u_list.append(uid)
        r_list.append(rid)
        ratings_list.append(ratings)

        Omega.append((uid, rid))

    f.close()

    # Build rating matrix A
    u_array = np.array(u_list)
    r_array = np.array(r_list)
    ratings_array = np.array(ratings_list)

    A = coo_matrix((ratings_array, (u_array, r_array)), shape=(m, n))

    if sparse:
        A = A.tocsr()
    else:
        A = A.toarray()

    print("Rating matrix loaded")

    return A, Omega

def divide_samples(Omega):

    cache = {}
    for i, j in Omega:
        if i not in cache:
            cache[i] = [j]
        else:
            cache[i].append(j)

    S_Omega = []
    T_Omega = []

    for i in cache.keys():
        if len(cache[i]) >= 3:
            j = random.choice(cache[i])
            cache[i].remove(j)
            T_Omega.append((i, j))

        if len(cache[i]) >= 2:
            j = random.choice(cache[i])
            cache[i].remove(j)
            T_Omega.append((i, j))            

        for j in cache[i]:
            S_Omega.append((i, j))

    return S_Omega, T_Omega


def main():
    # m = 478841
    m = 5000
    n = 26730

    sparse = False

    U_T_Prime_file = data_dir + 'U_prime_vector.p'
    V_T_Prime_file = data_dir + 'R_prime_vector.p'

    # Get the rating data
    # A, Omega = build_sparse_matrix(m, n)
    A, Omega = build_sparse_matrix(m, n, sparse=sparse)

    # Get U_T_Prime and V_T_Prime matrices
    U_T_Prime = pickle.load(open(U_T_Prime_file, "rb"))
    V_T_Prime = pickle.load(open(V_T_Prime_file, "rb"))

    if sparse:
        U_T_Prime = csr_matrix(U_T_Prime)
        V_T_Prime = csr_matrix(V_T_Prime)
    else:
        U_T_Prime = np.array(U_T_Prime)
        V_T_Prime = np.array(V_T_Prime)

    # number of dimension of features
    k = 5
    # k = 10

    k_prime = U_T_Prime.shape[1]
    if k > k_prime:
        # Append zero to make dismension consistent
        U_padding = np.zeros((U_T_Prime.shape[0], k - k_prime))
        V_padding = np.zeros((V_T_Prime.shape[0], k - k_prime))

        U_T_Prime = np.concatenate((U_T_Prime, U_padding), axis=1)
        V_T_Prime = np.concatenate((V_T_Prime, V_padding), axis=1)

    # Init U and V
    U_T = np.random.rand(m, k)
    V_T = np.random.rand(n, k)

    if sparse:
        U_T = csr_matrix(U_T)
        V_T = csr_matrix(V_T)

    # Dived the sample into training and testing set
    S_Omega, T_Omega = divide_samples(Omega)

    print(len(S_Omega))
    print(len(T_Omega))


    mse = MSE(A, U_T, V_T, T_Omega)
    # mse = distributed_MSE(A, U_T, V_T, Omega)
    print(mse)

    SGD(A, U_T, V_T, S_Omega, U_T_Prime, V_T_Prime)

    mse = MSE(A, U_T, V_T, T_Omega)
    # mse = distributed_MSE(A, U_T, V_T, Omega)
    print(mse)


if __name__ == '__main__':
    main()










