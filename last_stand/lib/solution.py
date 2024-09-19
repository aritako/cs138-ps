import os
import numpy as np
from mpmath import mp, matrix
mp.dps = 50
FOLDER_PATH = 'defense_plans'
# FOLDER_CATEGORIES = [
#   'balanced',
#   'classic',
#   'cold',
#   'hot',
#   'short',
#   'singko',
#   'straight',
#   'smol',
#   'tres',
#   'uno',
# ]
FOLDER_CATEGORIES = ['balanced']
NUM_TEST_CASES = 25
DAMAGE_MULTIPLIERS = {
  "G" : [0, 4, 0],
  "T" : [1, 1, 1],
  "R" : [0, 2, 0],
  "P" : [0, 1, 0],
  "W" : [0, 2, 0],
}
def main():
  for category in FOLDER_CATEGORIES:
    category_path = os.path.join(FOLDER_PATH, category)
    for i in range(0, NUM_TEST_CASES):
      # Assumption is num_test_case <= 25.
      file_name = f'0{i + 1}.txt' if len(str(i)) < 2 and i != 9 else f'{i + 1}.txt'
      file_path = os.path.join(category_path, file_name)
      
      if os.path.exists(file_path):
        with open(file_path, 'r') as file:
          defense_plan = file.read()
          print("Defense Plan Number: ", i + 1)
          protect_the_brains = defend_from_zombies(defense_plan)
          print(f"Solution: {protect_the_brains}\n")
          print("--------------------")

def defend_from_zombies(defense_plan):
  k, symbols, n, total = defense_plan.split('\n')
  damage_vector = compute_damage_vector(symbols)
  damage_matrix = build_damage_matrix(damage_vector, int(n))
  damage_total = [int(x) for x in total.split(' ')]
  solution = solve_damage_solution(damage_matrix, damage_total)
  [print(x) for x in solution]
  # print(damage_matrix)
  # print(damage_total)
  # print(solution)
  return [x for x in solution]

def solve_damage_solution(A, t):
  # Using mpmath's 'matrix' to solve the system of equations
  A = matrix(A)
  t = matrix(t)
  # Step 1: Perform Cholesky decomposition (A = L * L^T)
  L = mp.cholesky(A)
  
  # Step 2: Solve L * y = b using forward substitution
  y = mp.lu_solve(L, t)
  
  # Step 3: Solve L^T * x = y using backward substitution
  x = mp.lu_solve(L.T, y)
  
  return x

# def solve_damage_solution(damage_matrix, damage_total):
#   # Using mpmath's 'matrix' to solve the system of equations
#   A = np.array(damage_matrix, dtype=np.float64)
#   t = np.array(damage_total, dtype=np.float64)
#   try:
#       damage_solution = np.linalg.solve(A, t)
#   except Exception as e:
#       print(f"An error occurred during matrix solving: {e}")
#       return None
#   return damage_solution

def compute_damage_vector(symbols):
  damage_vector = [0, 0, 0]
  for i in symbols:
    if i == 'W':
      damage_vector[1] *= DAMAGE_MULTIPLIERS[i][1]
    else:
      damage_vector[0] += DAMAGE_MULTIPLIERS[i][0]
      damage_vector[1] += DAMAGE_MULTIPLIERS[i][1]
      damage_vector[2] += DAMAGE_MULTIPLIERS[i][2]
  for i in range(3):
    damage_vector[i] = mp.mpf(damage_vector[i])
  return damage_vector

def build_damage_matrix(damage_vector, n):
  damage_matrix = [[0 for _ in range(n)] for _ in range(n)]
  if n == 1:
    damage_matrix[0][0] = damage_vector[1]
    return damage_matrix
  for i in range(n):
    damage_matrix[i][i] = damage_vector[1]
    if i == 0:
      damage_matrix[i][i + 1] = damage_vector[2]
    elif i == n - 1:
      damage_matrix[i][i - 1] = damage_vector[0]
    else:
      damage_matrix[i][i - 1] = damage_vector[0]
      damage_matrix[i][i + 1] = damage_vector[2]
  for i in range(n):
    for j in range(n):
      damage_matrix[i][j] = mp.mpf(damage_matrix[i][j])
  return damage_matrix

main()