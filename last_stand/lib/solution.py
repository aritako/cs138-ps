import os
import numpy as np

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
FOLDER_CATEGORIES = ['classic']
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
    print("=====================================")
    print("DEFENSE NAME:", category)
    print("=====================================")
    for i in range(0, NUM_TEST_CASES):
      # Assumption is num_test_case <= 25.
      file_name = f'0{i + 1}.txt' if len(str(i)) < 2 and i != 9 else f'{i + 1}.txt'
      file_path = os.path.join(category_path, file_name)

      if os.path.exists(file_path):
        with open(file_path, 'r') as file:
          defense_plan = file.read()
          print("Defense Plan Number: ", i + 1)
          protect_the_brains = defend_from_zombies(defense_plan)
          print("--------------------")

def defend_from_zombies(defense_plan):
  k, symbols, n, total = defense_plan.split('\n')
  damage_vector = compute_damage_vector(symbols)
  # damage_matrix = build_damage_matrix(damage_vector, int(n))
  damage_total = [int(x) for x in total.split(' ')]
  # lower_diagonal, main_diagonal, upper_diagonal = generate_diagonals(damage_vector, int(n))
  # print(lower_diagonal, main_diagonal, upper_diagonal)
  solution = compute_solution(damage_vector, damage_total, int(n))
  # [print(x) for x in solution]
  # print(damage_matrix)
  # print(damage_total)
  # print(solution)
  return [x for x in solution]

def generate_diagonals(damage_vector, n):
  lower_diagonal = [damage_vector[0] for _ in range(n - 1)] if n > 1 else [0]
  main_diagonal = [damage_vector[1] for _ in range(n)]
  upper_diagonal = [damage_vector[2] for _ in range(n - 1)] if n > 1 else [0]
  print("")
  return lower_diagonal, main_diagonal, upper_diagonal

def compute_solution(damage_vector, b, n):
  print("Damage Vector: ", damage_vector)
  if n == 1:
    print("Special case: Single element matrix")
    print("Solution: ", round(b[0] / damage_vector[1]))
    return [round(b[0] / damage_vector[1])]
  if damage_vector[0] == 0 and damage_vector[2] == 0 and damage_vector[1] != 0:
    print("Special case: Zero adjacent diagonals")
    solution = round(b[0] / damage_vector[1]) 
    print("Solution: ", solution)

    return [solution] * n
  if damage_vector[0] == damage_vector[1] == damage_vector[2]:
    print("Special case: Equal diagonals")
    if damage_vector[0] == 0:
      print("Special case: All zero diagonals")
      return [0] * n
    return solve_equal_diagonals(n, damage_vector[1], damage_vector[0], b)
  # Forward elimination
  for i in range(n - 1):
    L = damage_vector[0] / damage_vector[1]   # Compute the multiplier
    damage_vector[1] -= L * damage_vector[2]  # Update the next diagonal element
    b[i + 1] -= L * b[i]  # Update the right-hand side vector
  
  x = [0] * n
  x[n - 1] = b[n - 1] / damage_vector[1]  # Solve for the last variable
  for i in range(n - 2, -1, -1):
      x[i] = (b[i] - damage_vector[2] * x[i + 1]) / damage_vector[1]  # Solve for the rest
  return x

def solve_equal_diagonals(n, d, s, b, epsilon=1e-10):
    # perturbation god i hope this works

    alpha = [0] * n
    beta = [0] * n 

    alpha[0] = d
    if abs(alpha[0]) < epsilon:
        alpha[0] += epsilon
    
    beta[0] = b[0] / alpha[0]

    for i in range(1, n):
        alpha[i] = d - (s ** 2) / alpha[i - 1]
        if abs(alpha[i]) < epsilon:
            alpha[i] += epsilon
        
        beta[i] = (b[i] - s * beta[i - 1]) / alpha[i]

    x = [0] * n
    x[n - 1] = beta[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = round(beta[i] - (s / alpha[i]) * x[i + 1])
    return x

def compute_damage_vector(symbols):
  damage_vector = [0, 0, 0]
  last_W_index = symbols.rfind('W')
  for idx, i in enumerate(symbols):
    if i == 'W' and idx != last_W_index:
      continue
    if i == 'W' and idx == last_W_index:
      damage_vector[1] *= DAMAGE_MULTIPLIERS[i][1]
    else:
      damage_vector[0] += DAMAGE_MULTIPLIERS[i][0]
      damage_vector[1] += DAMAGE_MULTIPLIERS[i][1]
      damage_vector[2] += DAMAGE_MULTIPLIERS[i][2]
  # for i in range(3):
  #   damage_vector[i] = mp.mpf(damage_vector[i])
  return damage_vector

def build_damage_matrix(damage_vector, n):
  damage_matrix = [[0 for _ in range(n)] for _ in range(n)]
  for i in range(n):
    damage_matrix[i][i] = damage_vector[1]
    if i > 0:
      damage_matrix[i][i - 1] = damage_vector[0]
    if i < n - 1:
      damage_matrix[i][i + 1] = damage_vector[2]
  return damage_matrix


main()