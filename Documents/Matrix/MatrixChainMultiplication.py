def print_optimal_parens(s, i, j):
    if i == j:
        return f"A{i+1}"
    else:
        return f"({print_optimal_parens(s, i, s[i][j])} × {print_optimal_parens(s, s[i][j]+1, j)})"


def matrix_multiplication(matrices):

    n = len(matrices)

    # validate input
    for i in range(n - 1):
        mat_a = matrices[i]
        mat_b = matrices[i + 1]
        cols_a = len(mat_a[0])
        rows_b = len(mat_b)
        if cols_a != rows_b:
            raise ValueError(f"Matrix {i} and Matrix {i+1} have incompatible dimensions: {cols_a} != {rows_b}")

    # 2D array to store minimum cost to compute matrix multiplication
    Cost = [[float('inf') for col in range(n)] for row in range(n)]
    # 2D array to store split point
    Split = [[-1 for _ in range(n)] for _ in range(n)]

    for i in range(len(matrices)):  # Base case
        Cost[i][i] = 0 

    for length in range(1,n+1):
        for i in range(n-length+1):
            j = i + length - 1
            for k in range(i,j):
                m_i = len(matrices[i]) # dimension of ith matrix
                m_j = len(matrices[j][0]) # dimension of jth matrix
                m_k = len(matrices[k][0])
                if Cost[i][j] > Cost[i][k]+Cost[k+1][j]+m_i*m_k*m_j: # Recurrence relation
                    Cost[i][j]= Cost[i][k]+Cost[k+1][j]+m_i*m_k*m_j
                    Split[i][j] = k

    optimal = print_optimal_parens(Split, 0, n - 1)
    print(f"Optimal Split: {optimal}")
    print(f"Minimum Cost: {Cost[0][n - 1]}")

    return Cost[0][n-1]
