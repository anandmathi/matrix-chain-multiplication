def print_optimal_parens(s, i, j):
    if i == j:
        return f"A{i+1}"
    else:
        return f"({print_optimal_parens(s, i, s[i][j])} × {print_optimal_parens(s, s[i][j]+1, j)})"


def matrix_multiplication(matrices):

    n = len(matrices)
    #2D array to store minimum cost to compute matrix multiplication
    
    Cost = [[0 for col in range(n)] for row in range(n)]

    for i in range(len(matrices)):  #Base case
        Cost[i][i] = 0 

    for length in range(1,n+1):
        for i in range(1,n-length+1):
            j = i+length -1
            for k in range(i,j):
                m_i = len(matrices[i-1]) # dimension of ith matrix
                m_j = len(matrices[j][0]) # dimension of jth matrix
                m_k = len(matrices[k][0])
                if Cost[i][j] > Cost[i][k]+Cost[k+1][j]+m_i*m_k*m_j: #Recurrence relation
                    Cost[i][j]= Cost[i][k]+Cost[k+1][j]+m_i*m_k*m_j

    return Cost[0][n-1]