
import pickle
import csv

from scipy.sparse import *
import numpy as np
import random

# data_dir = './5000/'
data_dir = './'

def spark_build_ones_block_matrix(m, partition_num):
    ones = np.ones((m, 1))

    m_partition = m / partition_num

    blocks = []
    for i in range(partition_num):
        block_index = (i, 0)
        selected_array = ones[i*m_partition:(i+1)*m_partition, :]
        # print selected_array
        # print selected_array.shape
        selected_array = selected_array.flatten()
        # print i
        # print selected_array
        # print selected_array.shape
        block_array = Matrices.dense(m_partition, 1, selected_array)
        blocks.append((block_index, block_array))


    blocks = sc.parallelize(blocks)
    # Create a BlockMatrix from an RDD of sub-matrix blocks.
    mat_m = BlockMatrix(blocks, m_partition, 1)
    m = mat_m.numRows() # 6
    n = mat_m.numCols() # 2
    # print 'in ones'
    # print 'rows: ', m
    # print 'cols: ', n
    return mat_m

def spark_build_rating_matrix(m, n, partition_num):
    m_partition = m / partition_num
    n_partition = n / partition_num

    total_blocks = []
    block = np.zeros((m_partition, n))
    partition_index = 0
    ratings_csv = '../../data/preprocessed_data/ratings.csv'

    with open(ratings_csv, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            # user id
            uid = int(row[0])
            # print 'uid: ', uid
            if uid >= (partition_index+1)*m_partition:
                # print 'in partition: ', uid
                # split block on column direction
                for i in range(partition_num):
                    block_index = (partition_index, i)
                    selected_array = block[:, i*n_partition: (i+1)*n_partition]

                    # construct sparse matrix
                    row_indexes = []
                    col_indexes = []
                    for column in range(n_partition):
                        column_array = selected_array[:, column]
                        non_zero_rows = column_array.nonzero()[0]
                        col_indexes.append(len(row_indexes))
                        row_indexes += non_zero_rows.tolist()

                    # need one more
                    col_indexes.append(len(row_indexes))
                    nonzeros = selected_array.transpose()[selected_array.transpose().nonzero()]
                    # use sparse matrix to avoid memory overflow
                    block_array = Matrices.sparse(m_partition, n_partition, col_indexes, row_indexes, nonzeros)
                    total_blocks.append((block_index, block_array))
                    
                # reinitialize block
                block = np.zeros((m_partition, n))
                # update partition_index
                partition_index += 1

                if uid >= m:
                    # print 'hhhhh'
                    break

            rid = int(row[1])
            uid = uid - partition_index*m_partition
            ratings = float(row[2])
            block[uid, rid] = ratings

    # print 'total_blocks: ', len(total_blocks)
    # create block matrix
    blocks = sc.parallelize(total_blocks)
    # Create a BlockMatrix from an RDD of sub-matrix blocks.
    result = BlockMatrix(blocks, m_partition, n_partition)
    m = result.numRows() 
    n = result.numCols() 
    # print 'rows: ', m
    # print 'cols: ', n
    m = result.numRowBlocks
    n = result.numColBlocks
    # print 'row blocks: ', m
    # print 'col blocks: ', n
    return result

def spark_matrix_multiply(m, n, k, partition_num):
    U_T = np.random.rand(m, k)
    V_T = np.random.rand(n, k)

    m_partition = m / partition_num
    n_partition = n / partition_num

    m_blocks = []
    for i in range(partition_num):
        block_index = (i, 0)
        selected_array = U_T[i*m_partition:(i+1)*m_partition, :]
        # print selected_array
        # print selected_array.shape
        selected_array = selected_array.flatten()
        # print i
        # print selected_array
        # print selected_array.shape
        block_array = Matrices.dense(m_partition, k, selected_array)
        m_blocks.append((block_index, block_array))


    blocks = sc.parallelize(m_blocks)
    # Create a BlockMatrix from an RDD of sub-matrix blocks.
    mat_m = BlockMatrix(blocks, m_partition, k)
    m = mat_m.numRows() # 6
    n = mat_m.numCols() # 2
    # print 'rows: ', m
    # print 'cols: ', n

    # for V_T
    n_blocks = []
    for i in range(partition_num):
        block_index = (i, 0)
        selected_array = V_T[i*n_partition:(i+1)*n_partition, :]
        # print selected_array
        # print selected_array.shape
        selected_array = selected_array.flatten()
        # print i
        # print selected_array
        # print selected_array.shape
        block_array = Matrices.dense(n_partition, k, selected_array)
        n_blocks.append((block_index, block_array))


    blocks = sc.parallelize(n_blocks)
    # Create a BlockMatrix from an RDD of sub-matrix blocks.
    mat_n = BlockMatrix(blocks, n_partition, k)
    mat_n = mat_n.transpose()
    m = mat_n.numRows() # 6
    n = mat_n.numCols() # 2
    # print 'rows: ', m
    # print 'cols: ', n
    # 
    
    # multiply U_T and V_T
    result = mat_m.multiply(mat_n)
    m = result.numRowBlocks
    n = result.numColBlocks
    # print 'rows: ', m
    # print 'cols: ', n
    
    return result


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

    # mse_list = []

    for _ in range(100):
        # One pass
        random.shuffle(Omega)

        for (i, j) in Omega:
            tmp = np.dot(U_T[i], V_T[j].T) - A[i, j]

            U_T[i] = U_T[i] - step_size * (2 * tmp * V_T[j] - 1 * V_T_Prime[j])
            V_T[j] = V_T[j] - step_size * (2 * tmp * U_T[i] - 1 * U_T_Prime[i])

            # U_T[i] = U_T[i] - step_size * 2 * tmp * V_T[j]
            # V_T[j] = V_T[j] - step_size * 2 * tmp * U_T[i]

    #     mse = MSE(A, U_T, V_T, Omega)
    #     mse_list.append(mse)

    # for mse in mse_list:
    #     print(mse)



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
    m = 478841
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


    # mse = MSE(A, U_T, V_T, T_Omega)
    mse = distributed_MSE(A, U_T, V_T, T_Omega)
    print(mse)

    SGD(A, U_T, V_T, S_Omega, U_T_Prime, V_T_Prime)

    # mse = MSE(A, U_T, V_T, T_Omega)
    mse = distributed_MSE(A, U_T, V_T, T_Omega)
    print(mse)


if __name__ == '__main__':
    main()










