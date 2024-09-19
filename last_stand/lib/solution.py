import os
import glob

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
FOLDER_CATEGORIES = ['sample']
NUM_TEST_CASES = 1

def main():
  for category in FOLDER_CATEGORIES:
    category_path = os.path.join(FOLDER_PATH, category)
    for i in range(0, NUM_TEST_CASES):
      # Assumption is num_test_case <= 25.
      file_name = f'0{i + 1}.txt' if len(str(i)) < 2 and i != 9 else f'{i + 1}.txt'
      print(file_name)
      file_path = os.path.join(category_path, file_name)
      
      if os.path.exists(file_path):
        with open(file_path, 'r') as file:
          defense_plan = file.read()
          print(f'Contents of {file_path}:')
          # print(content)
          print('---')
  k, symbols, n, total = defense_plan.split('\n')
  
main()