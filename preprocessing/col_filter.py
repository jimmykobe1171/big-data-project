
import pickle
import csv

from scipy.sparse import *
import numpy as np
import random
from pprint import pprint
from copy import copy

def MSE(A, U_T, V_T):
    mse = 0

    Ahat = np.dot(U_T, V_T.T)
    m, n = A.shape

    return 0

    # ct = 0
    # for i in range(m):
    #     for j in range(n):
    #         if A[i][j] != 0:
    #             ct += 1
    #             mse += abs(A[i][j] - Ahat[i][j])
    
    # return mse / ct

def SGD(A, U_T, V_T, Omega, U_T_Prime, V_T_Prime):
    """
    :param:
        A: csr_matrix
        U_T: np.array
        V_T: np.array
    """
    step_size = 0.001

    for _ in range(100):
        # One pass
        local_omega = copy(Omega)
        random.shuffle(local_omega)

        while local_omega:
            i, j = local_omega.pop()

            tmp = (np.dot(U_T[i], V_T[j].T) - A[i][j])

            U_T[i] = U_T[i] - step_size * (2 * tmp * V_T[j].T - V_T_Prime[j].T)

            V_T[j] = V_T[j] - step_size * (2 * tmp * U_T[i] - U_T_Prime[i])


def build_sparse_matrix(m, n):
    ratings_csv = 'ratings.csv'

    # A = np.zeros((m, n))
    # A = csr_matrix((m, n), dtype=np.int8)

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

    A = coo_matrix((ratings_array, (u_array, r_array)), shape=(m, n)).tocsr()

    print("Rating matrix loaded")
    print(A.shape)

    return A, Omega

def matrix_multiply(U_T, V_T):

    m, k = U_T.shape

    cache = []


    




def main():
    m = 478841
    n = 26730

    # U_T_Prime_file = 'U_prime_vector.p'
    # V_T_Prime_file = 'R_prime_vector.p'

    # # Get the rating data
    # A, Omega = build_sparse_matrix(m, n)

    # # Get U_T_Prime and V_T_Prime matrices
    # U_T_Prime = pickle.load(open(U_T_Prime_file, "rb"))
    # V_T_Prime = pickle.load(open(V_T_Prime_file, "rb"))

    # number of dimension of features
    k = 5 

    # Init U and V
    # U_T = np.random.rand(m, k)
    # V_T = np.random.rand(n, k)

    U_T = csr_matrix(np.random.rand(m, k))
    V_T = csr_matrix(np.random.rand(n, k))

    print(U_T.shape)
    print(V_T.shape)


    matrix_multiply(U_T, V_T)



    # mse = MSE(A, U_T, V_T)
    # print(mse)

    # SGD(A, U_T, V_T, Omega, U_T_Prime, V_T_Prime)

    # mse = MSE(A, U_T, V_T)
    # print(mse)


if __name__ == '__main__':
    main()










