import math

def first_equation(w_2, w_3):
    return -(w_2*math.e**-4) - (w_3*math.e**-16) + 1

def second_equation(w_1, w_3):
    return (55/9)*(math.e**-4)*(w_1 + w_3)

def third_equation(w_1, w_2):
    return -(w_1*math.e**-16) - (w_2*math.e**-4) + 1

def jacobi_method(iterations):
    initial_guess = [1, 1, 1]
    w_1, w_2, w_3 = initial_guess
    for i in range(iterations):
        w_1 = first_equation(w_2, w_3)
        w_2 = second_equation(w_1, w_3)
        w_3 = third_equation(w_1, w_2)
        print(f'Iteration {i+1}')
        print(f'w_1 = {w_1}')
        print(f'w_2 = {w_2}')
        print(f'w_3 = {w_3}')
        print("-------------------")
    return w_1, w_2, w_3

jacobi_method(3)