import os
import random
import copy

def parse_input(filename):
    garden_dir = os.path.join(os.path.dirname(__file__), '../gardens', filename)
    
    with open(garden_dir, 'r') as f:
        n = int(f.readline().strip())
        matrix = []
        for _ in range(n):
            row = list(map(int, f.readline().strip().split()))
            matrix.append(row)
    return matrix

def write_matrix_to_file(matrix, filename):
    garden_dir = os.path.join(os.path.dirname(__file__), '../gardens', filename)

    with open(garden_dir, 'w') as f:
        f.write(f"{len(matrix)}\n")
        for row in matrix:
            f.write(" ".join(map(str, row)) + "\n")

def matrix_vector_multiply(matrix, s):
    product = []
    for row in matrix:
        row_sum = sum([row[j] * s[j] for j in range(len(s))])
        product.append(row_sum)
    return product

def print_matrix(matrix):
    for row in matrix:
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
    norm_matrix = inf_norm(matrix)
    matrix_inv = inverse(matrix)
    norm_matrix_inv = inf_norm(matrix_inv)

    return norm_matrix * norm_matrix_inv

def gaussian_elimination_with_partial_pivoting(matrix, b):
    n = len(matrix)
    
    # Forward elimination with partial pivoting
    for i in range(n):
        # Find the maximum element in the current column to use as a pivot
        max_row = max(range(i, n), key=lambda k: abs(matrix[k][i]))
        if matrix[max_row][i] == 0:
            raise ValueError("Matrix is singular or nearly singular")
        
        # Swap rows if needed
        matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
        b[i], b[max_row] = b[max_row], b[i]
        
        # Normalize the pivot row
        pivot = matrix[i][i]
        matrix[i] = [matrix[i][j] / pivot for j in range(n)]
        b[i] /= pivot
        
        # Eliminate the current column in the rows below
        for k in range(i+1, n):
            factor = matrix[k][i]
            matrix[k] = [matrix[k][j] - factor * matrix[i][j] for j in range(n)]
            b[k] -= factor * b[i]
    
    # Backward substitution
    s_hat = [0] * n
    for i in range(n-1, -1, -1):
        s_hat[i] = b[i] - sum(matrix[i][j] * s_hat[j] for j in range(i+1, n))
    
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

gardens = ['A.txt', 'B.txt', 'C.txt', 'D.txt', 'E.txt', 'F.txt', 'G.txt', 'H.txt', 'I.txt', 'J.txt']

# Z = parse_input('A.txt')
# print(matrix_vector_multiply(Z, [1, 1, 1]))
# for garden in gardens:
#     Z = parse_input(garden)
#     Z1, Z2 = generate_unique_rearrangements(Z)
    
#     base_filename = garden.split('.')[0]
#     write_matrix_to_file(Z1, f"{base_filename}1.txt")
#     write_matrix_to_file(Z2, f"{base_filename}2.txt")

#     print(f"Processed {garden} and created rearrangements")

# Use an absolute path for the gardens directory
gardens_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gardens'))

# List the files in the gardens directory
gardens = [f for f in os.listdir(gardens_dir) if f.endswith('.txt')]

for garden in gardens:
    garden_path = os.path.join(gardens_dir, garden)
    Z = parse_input(garden)
    n = len(Z)
    s = [1] * n
    b = matrix_vector_multiply(Z, s)

    Z_copy = copy.deepcopy(Z)
    b_copy = copy.deepcopy(b)
    
    s_hat = gaussian_elimination_with_partial_pivoting(Z_copy, b_copy)
    inf_norm_delta_s = inf_norm(matrix_subtraction(s, s_hat))
    Z_condition_number = condition_number(Z)
    # Append the computed values to the original file
    with open(garden_path, 'a') as f:
        f.write(" ".join(map(str, b)) + "\n")
        f.write(" ".join(map(str, s_hat)) + "\n")
        f.write(f"{inf_norm_delta_s}\n")
        f.write(f"{Z_condition_number}")