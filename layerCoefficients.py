import numpy as np

def layerCoefficients(numberOfLayers, poisson, lambda_i, subtractLambdas, R, m):

    F = np.append(np.exp(-m*subtractLambdas),0)
    
    # Matrix first layer z=0 sigma_z=-mJ0(m*distanceRadial) y tau_rz=0
    x_1 = np.array([[np.exp(-m*lambda_i[0]),    1, np.exp(-m*lambda_i[0])*(2*poisson[0]-1),    1-2*poisson[0]],
                    [np.exp(-m*lambda_i[0]),    -1, np.exp(-m*lambda_i[0])*(2*poisson[0]),     2*poisson[0]]])

    # List of coefficients from the second layer to the last layer
    rangeStop = numberOfLayers - 1
    unknownCoefficients = numberOfLayers * 4 - 2
    M = [np.zeros((4,4)) for i in range(rangeStop)]
    N = [np.zeros((4,4)) for i in range(rangeStop)]

    for i in range(rangeStop):
        
        M[i] = np.array([   [1,     F[i],   - 1 + 2 * poisson[i] + m * lambda_i[i],  (1 - 2 * poisson[i] + m * lambda_i[i]) * F[i]], 
                            [1,     -F[i],  2 * poisson[i] + m * lambda_i[i],        (2 * poisson[i] - m * lambda_i[i]) * F[i]], 
                            [1,     F[i],   1 + m * lambda_i[i],                     (-1 + m * lambda_i[i]) * F[i]], 
                            [1,     -F[i],  - 2 + 4 * poisson[i] + m * lambda_i[i],  (- 2 + 4 * poisson[i] - m * lambda_i[i]) * F[i]]])
        
        if i < rangeStop:
            N[i] = np.array([   [F[i+1],            1,      (- 1 + 2 * poisson[i+1] + m * lambda_i[i]) * F[i+1],         1 - 2 * poisson[i+1] + m * lambda_i[i]],
                                [F[i+1],            -1,     (2 * poisson[i+1] + m * lambda_i[i]) * F[i+1],               2 * poisson[i+1] - m * lambda_i[i]],
                                [R[i] * F[i+1],     R[i],   (1 + m * lambda_i[i]) * R[i] * F[i+1],                       (-1 + m * lambda_i[i]) * R[i]],
                                [R[i] * F[i+1],     -R[i],  (- 2 + 4 * poisson[i+1] + m * lambda_i[i]) * R[i] * F[i+1],  (- 2 + 4 * poisson[i+1] - m * lambda_i[i]) * R[i]]])

    globalArray = np.array(np.zeros((unknownCoefficients, unknownCoefficients + 2)))
    globalArray[0:2, 0:4] = x_1

    for i, array in enumerate(M):
        position = i * 4 + 2
        globalArray[position:position + 4,position-2:position + 2] = M[i]
        globalArray[position:position + 4, position+2:position + 6] = N[i]

    globalArray = np.delete(globalArray, [-2, -4], axis=1)
    resultsVector = np.array(np.zeros((unknownCoefficients, 1)))
    resultsVector[0] = 1
    coefficients = np.linalg.solve(globalArray, resultsVector)
    
    return coefficients