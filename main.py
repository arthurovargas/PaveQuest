import time

from PaveQuest import PavementResponses

init = time.time()

thickness = [6, 6]
module = [400000, 20000, 10000]
poisson = [0.5, 0.5, 0.5]
m = 50

[coefficients_a, coefficients_b, coefficients_c, coefficients_d] = PavementResponses.layer_coefficients(module, poisson,
                                                                                                        thickness, m)
fin = time.time()
print(fin - init)

print(coefficients_a, coefficients_b, coefficients_c, coefficients_d)