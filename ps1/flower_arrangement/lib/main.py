import os
import random
import sys

def parse_input(filename):
    garden_dir = os.path.join(os.path.dirname(__file__), '../gardens', filename)
    
    with open(garden_dir, 'r') as f:
        n = int(f.readline().strip())
        Z = []
        for _ in range(n):
            row = list(map(int, f.readline().strip().split()))
            Z.append(row)
    return Z

def write_matrix_to_file(matrix, filename):
    garden_dir = os.path.join(os.path.dirname(__file__), '../gardens', filename)

    with open(garden_dir, 'w') as f:
        f.write(f"{len(matrix)}\n")
        for row in matrix:
            f.write(" ".join(map(str, row)) + "\n")

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
    # Check if it's a 1D array (vector)
    if isinstance(matrix[0], (int, float)):
        # Infinity norm of a vector: max absolute value of elements
        return max(abs(element) for element in matrix)
    
    # Otherwise, it's a 2D array (matrix)
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

def is_diagonal(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i != j and matrix[i][j] != 0:
                return False
    return True

def determinant(matrix):
    n = len(matrix)
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    # Make a copy of the matrix to avoid modifying the original
    mat = [row[:] for row in matrix]
    
    det = 1
    for i in range(n):
        # Find the pivot element
        pivot = mat[i][i]
        if pivot == 0:
            # Find a row with a non-zero element in the same column
            for j in range(i + 1, n):
                if mat[j][i] != 0:
                    # Swap rows
                    mat[i], mat[j] = mat[j], mat[i]
                    pivot = mat[i][i]
                    det *= -1  # Swapping rows changes the sign of the determinant
                    break
            else:
                # If no non-zero pivot is found, the matrix is singular
                return 0
        
        # Scale the pivot row
        det *= pivot
        for j in range(i, n):
            mat[i][j] /= pivot
        
        # Eliminate the current column in rows below the pivot
        for j in range(i + 1, n):
            factor = mat[j][i]
            for k in range(i, n):
                mat[j][k] -= factor * mat[i][k]
    
    return det

# Helper function to calculate the minor of a matrix (remove a row and a column)
def minor(matrix, row, col):
    return [row[:col] + row[col + 1:] for row in (matrix[:row] + matrix[row + 1:])]

def swap_rows(matrix, row1, row2):
    matrix[row1], matrix[row2] = matrix[row2], matrix[row1]
    return matrix

def flatten_matrix(matrix):
    return [element for row in matrix for element in row]

def reshape_list_to_matrix(lst, rows, cols):
    return [lst[i * cols:(i + 1) * cols] for i in range(rows)]

def generate_unique_rearrangements(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    
    if is_diagonal(matrix):
        # Extract the non-zero diagonal elements
        diagonal_elements = [matrix[i][i] for i in range(rows)]
        
        while True:
            # Generate first unique rearrangement of diagonal elements
            random.shuffle(diagonal_elements)
            rearrangement1 = [[0] * cols for _ in range(rows)]
            for i in range(rows):
                rearrangement1[i][i] = diagonal_elements[i]
            if determinant(rearrangement1) != 0:
                break
        
        while True:
            # Generate second unique rearrangement of diagonal elements
            random.shuffle(diagonal_elements)
            rearrangement2 = [[0] * cols for _ in range(rows)]
            for i in range(rows):
                rearrangement2[i][i] = diagonal_elements[i]
            if determinant(rearrangement2) != 0:
                break
        
        return rearrangement1, rearrangement2
    
    else:
        # Flatten the matrix
        flat_matrix = flatten_matrix(matrix)
        
        while True:
            # Generate first unique rearrangement
            random.shuffle(flat_matrix)
            rearrangement1 = reshape_list_to_matrix(flat_matrix, rows, cols)
            if determinant(rearrangement1) != 0:
                break
        
        while True:
            # Generate second unique rearrangement
            random.shuffle(flat_matrix)
            rearrangement2 = reshape_list_to_matrix(flat_matrix, rows, cols)
            if determinant(rearrangement2) != 0:
                break
        
        return rearrangement1, rearrangement2

def matrix_subtraction(A, B):
    # Case 1: If both are 1D arrays (vectors)
    if isinstance(A[0], (int, float)) and isinstance(B[0], (int, float)):
        if len(A) != len(B):
            raise ValueError("Vectors must have the same length for subtraction.")
        return [A[i] - B[i] for i in range(len(A))]

    # Case 2: Both are 2D matrices
    elif len(A) == len(B) and len(A[0]) == len(B[0]):
        # Create a result matrix with the same dimensions as A and B
        C = [[0 for _ in range(len(A[0]))] for _ in range(len(A))]

        # Perform element-wise subtraction for matrices
        for i in range(len(A)):
            for j in range(len(A[0])):
                C[i][j] = A[i][j] - B[i][j]

        return C

    else:
        raise ValueError("Matrices must have the same dimensions for subtraction.")

# Make sure to define a different s_hat! Right now, s_hat should be equal to s for checking. 
def process_garden(matrix):
    Z = matrix
    n = len(Z)

    print(f"Matrix Z of n {n}:")
    print_matrix(Z)

    s = [1] * n

    print("\nSolving for b, where Zs = b:")
    b = matrix_vector_multiply(Z, s)
    print(b)

    print("\nSolving for s")
    s_hat = gaussian_elimination_with_partial_pivoting(Z, b)
    print(s_hat)

    print("\nComputing the inf-norm of delta s = s_hat - s")
    inf_norm_delta_s = inf_norm(matrix_subtraction(s_hat, s))
    print(inf_norm_delta_s)

    print("\nComputing the condition number of Z under the inf-norm")
    Z_condition_number = condition_number(Z)
    print(Z_condition_number)
    print('\n')

gardens = ['A.txt', 'B.txt', 'C.txt', 'D.txt', 'E.txt', 'F.txt', 'G.txt', 'H.txt', 'I.txt', 'J.txt']

# The first and second rearrangement's results are located with their respective original matrix results.
for garden in gardens:
    Z = parse_input(garden)
    Z1, Z2 = generate_unique_rearrangements(Z)
    
    base_filename = garden.split('.')[0]
    write_matrix_to_file(Z1, f"{base_filename}1.txt")
    write_matrix_to_file(Z2, f"{base_filename}2.txt")

    log_filename = f"{base_filename}_log.txt"

    lib_dir = os.path.join(os.path.dirname(__file__), '../lib', log_filename)
    with open(lib_dir, 'w') as log_file:
        # Redirect stdout to the log file
        original_stdout = sys.stdout
        sys.stdout = log_file
        
        try:
            process_garden(Z)
            process_garden(Z1)
            process_garden(Z2)
        finally:
            # Restore stdout
            sys.stdout = original_stdout

    print(f"Processed {garden} and wrote log to {log_filename}")