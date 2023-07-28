import numpy as np


class PavementResponses:

    @staticmethod
    def layer_coefficients(module: list, poisson: list, thickness: list, m: int) -> list:

        module = np.array(module)
        poisson = np.array(poisson)
        thickness = np.array(thickness)

        number_of_layers = len(module)
        lambda_i = np.cumsum(thickness) / sum(thickness)  # lambda = zi/H
        R = module[0:-1] / module[1:] * ((1+poisson[1:]) / (1 + poisson[0:-1]))  # Known as R in the books

        subtract_lambdas = np.concatenate(([lambda_i[0]], np.diff(lambda_i)))  # e^(-m(lambda_i - lambda_i-1))
        F = np.append(np.exp(-m * subtract_lambdas), 0)

        # First layer array. z=0 sigma_z=-mJ0(m*distance_radial) y tau_rz=0
        first_layer_array = np.array(
            [[np.exp(-m * lambda_i[0]), 1, np.exp(-m * lambda_i[0]) * (2 * poisson[0] - 1), 1 - 2 * poisson[0]],
             [np.exp(-m * lambda_i[0]), -1, np.exp(-m * lambda_i[0]) * (2 * poisson[0]), 2 * poisson[0]]])

        # List of coefficients from the second layer to the last layer
        range_stop = number_of_layers - 1
        unknown_coefficients = number_of_layers * 4 - 2
        left_array = [np.zeros((4, 4)) for i in range(range_stop)]
        right_array = [np.zeros((4, 4)) for i in range(range_stop)]

        for i in range(range_stop):

            left_array[i] = np.array(
                [[1, F[i], - 1 + 2 * poisson[i] + m * lambda_i[i],
                  (1 - 2 * poisson[i] + m * lambda_i[i]) * F[i]],
                 [1, -F[i], 2 * poisson[i] + m * lambda_i[i],
                  (2 * poisson[i] - m * lambda_i[i]) * F[i]],
                 [1, F[i], 1 + m * lambda_i[i], (-1 + m * lambda_i[i]) * F[i]],
                 [1, -F[i], - 2 + 4 * poisson[i] + m * lambda_i[i],
                  (- 2 + 4 * poisson[i] - m * lambda_i[i]) * F[i]]])

            if i < range_stop:
                right_array[i] = np.array(
                    [[F[i + 1], 1, (- 1 + 2 * poisson[i + 1] + m * lambda_i[i]) * F[i + 1],
                      1 - 2 * poisson[i + 1] + m * lambda_i[i]],
                     [F[i + 1], -1, (2 * poisson[i + 1] + m * lambda_i[i]) * F[i + 1],
                      2 * poisson[i + 1] - m * lambda_i[i]],
                     [R[i] * F[i + 1], R[i], (1 + m * lambda_i[i]) * R[i] * F[i + 1],
                      (-1 + m * lambda_i[i]) * R[i]],
                     [R[i] * F[i + 1], -R[i],
                      (- 2 + 4 * poisson[i + 1] + m * lambda_i[i]) * R[i] * F[i + 1],
                      (- 2 + 4 * poisson[i + 1] - m * lambda_i[i]) * R[i]]])

        global_array = np.array(np.zeros((unknown_coefficients, unknown_coefficients + 2)))
        global_array[0:2, 0:4] = first_layer_array

        for i in range(len(left_array)):
            position = i * 4 + 2
            global_array[position:position + 4, position - 2:position + 2] = left_array[i]
            global_array[position:position + 4, position + 2:position + 6] = right_array[i]

        global_array = np.delete(global_array, [-2, -4], axis=1)
        results_vector = np.array(np.zeros((unknown_coefficients, 1)))
        results_vector[0] = 1
        coefficients = np.linalg.solve(global_array, results_vector)
        coefficients = np.insert(coefficients, (-1, -2), 0)

        coefficients_a = coefficients[::4]
        number_of_coefficients = number_of_layers * 4
        coefficients_b = coefficients[range(1, number_of_coefficients, 4)]
        coefficients_c = coefficients[range(2, number_of_coefficients, 4)]
        coefficients_d = coefficients[range(3, number_of_coefficients, 4)]

        return [coefficients_a, coefficients_b, coefficients_c, coefficients_d]
