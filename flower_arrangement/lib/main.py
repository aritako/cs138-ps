import os

def parse_input(filename):
    garden_dir = os.path.join(os.path.dirname(__file__), '../gardens', filename)
    
    with open(garden_dir, 'r') as f:
        n = int(f.readline().strip())
        Z = []
        for _ in range(n):
            row = list(map(int, f.readline().strip().split()))
            Z.append(row)
    return Z

def matrix_vector_multiply(Z, s):
    b = []
    for row in Z:
        row_sum = sum([row[j] * s[j] for j in range(len(s))])
        b.append(row_sum)
    return b

def print_matrix(Z):
    for row in Z:
        print(" ".join(map(str, row)))

def inf_norm(matrix):
    return max(sum(abs(element) for element in row) for row in matrix)

def inverse(matrix):
    n = len(matrix)
    augmented_matrix = [row[:] + [float(i==j) for i in range(n)] for j, row in enumerate(matrix)]

    for i in range(n):
        max_row = max(range(i, n), key=lambda k: abs(augmented_matrix[k][i]))
        augmented_matrix[i], augmented_matrix[max_row] = augmented_matrix[max_row], augmented_matrix[i]

        pivot = augmented_matrix[i][i]
        augmented_matrix[i] = [x / pivot for x in augmented_matrix[i]]

        for j in range(n):
            if j != i:
                factor = augmented_matrix[j][i]
                augmented_matrix[j] = [augmented_matrix[j][k] - factor * augmented_matrix[i][k] for k in range(2*n)]

    return [row[n:] for row in augmented_matrix]

def condition_number(matrix):
    norm_Z = inf_norm(matrix)
    Z_inv = inverse(matrix)
    norm_Z_inv = inf_norm(Z_inv)

    return norm_Z * norm_Z_inv

def gaussian_elimination_with_partial_pivoting(Z, b):
    n = len(Z)
    
    # Forward elimination with partial pivoting
    for i in range(n):
        # Find the maximum element in the current column to use as a pivot
        max_row = max(range(i, n), key=lambda k: abs(Z[k][i]))
        if Z[max_row][i] == 0:
            raise ValueError("Matrix is singular or nearly singular")
        
        # Swap rows if needed
        Z[i], Z[max_row] = Z[max_row], Z[i]
        b[i], b[max_row] = b[max_row], b[i]
        
        # Normalize the pivot row
        pivot = Z[i][i]
        Z[i] = [Z[i][j] / pivot for j in range(n)]
        b[i] /= pivot
        
        # Eliminate the current column in the rows below
        for k in range(i+1, n):
            factor = Z[k][i]
            Z[k] = [Z[k][j] - factor * Z[i][j] for j in range(n)]
            b[k] -= factor * b[i]
    
    # Backward substitution
    s_hat = [0] * n
    for i in range(n-1, -1, -1):
        s_hat[i] = b[i] - sum(Z[i][j] * s_hat[j] for j in range(i+1, n))
    
    return s_hat


def gauss_seidel(Z, b, tol=1e-6, max_iterations=1000):
    n = len(Z)
    s_hat = [0] * n  # Initial guess
    
    for _ in range(max_iterations):
        s_new = s_hat[:]
        for i in range(n):
            sum_before = sum(Z[i][j] * s_new[j] for j in range(i))
            sum_after = sum(Z[i][j] * s_hat[j] for j in range(i + 1, n))
            s_new[i] = (b[i] - sum_before - sum_after) / Z[i][i]
        
        # Check for convergence
        if max(abs(s_new[i] - s_hat[i]) for i in range(n)) < tol:
            return s_new
        
        s_hat = s_new
    
    raise ValueError("Gauss-Seidel did not converge")

def solve_system(Z, b):
    # Calculate the condition number
    cond_Z = condition_number(Z)
    print(f"Condition number: {cond_Z}")
    
    # Threshold to decide between LU Decomposition and Gauss-Seidel
    threshold = 1000
    
    if cond_Z < threshold:
        print("Using Gaussian elimination with partial pivoting")
        s_hat = gaussian_elimination_with_partial_pivoting(Z, b)
    else:
        print("Using Gauss-Seidel iteration")
        s_hat = gauss_seidel(Z, b)
    
    return s_hat

def process_garden(filename):
    Z = parse_input(filename)
    n = len(Z)

    print(f"Matrix {filename} of n {n}:")
    print_matrix(Z)

    s = [1] * n

    print("\nSolving for b, where Zs = b:")
    b = matrix_vector_multiply(Z, s)
    print(b)

    s_hat = solve_system(Z, b)
    print(s_hat)

gardens = ['A.txt', 'B.txt', 'C.txt', 'D.txt', 'E.txt', 'F.txt', 'G.txt', 'H.txt', 'I.txt', 'J.txt']

process_garden(gardens[0])