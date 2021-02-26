import numpy as np
import cvxpy as cp
import math
import time

from sklearn.preprocessing import PolynomialFeatures
from utils import addNames, readOptions, printSeparator, load_array

def PSR_lasso(x_points, y_points, numMeasurement, alpha, degree, nMono, N, t, learnedFeature, maskList):

    optionNames = readOptions()
    assert (len(optionNames) == N)

    extendedNames = []          # This list stores the name of the features, including vanilla features and low degree features
    featureIDList = []          # This list stores the variable IDs of each feature

    def getBasisValue(input, basis, featureIDList):  # Return the value of a basis
        ans = 0
        for (weight, pos) in basis:  # for every term in the basis
            term = weight
            for entry in featureIDList[pos]:  # for every variable in the term
                term = term * input[entry]
            ans += term
        return ans

    def get_features(x, degree):
        print("Extending feature vectors with degree " + str(degree) + " ..")
        featureExtender = PolynomialFeatures(degree,
                                             interaction_only=True)  # This function generates low degree monomials. It's a little bit slow though. You may write your own function using recursion.
        tmp = []
        for current_x in x:
            tmp.append(featureExtender.fit_transform(np.array(current_x).reshape(1, -1))[
                           0].tolist())  # Extend feature vectors with monomials
        return tmp


    for depth in range(0, degree+1):
        addNames('', 0, depth, 0, [], optionNames, extendedNames, featureIDList, N)

    x_points = x_points.reshape((numMeasurement, N))
    y_points = y_points.reshape(numMeasurement)

    # Removing NaN values on the measurement.
    inds = np.where(np.isnan(y_points))
    y_points[inds] = np.nanmax(y_points)

    x_points = np.array(get_features(x_points, degree))


    # Calculating the Lasso:
    numFeatures = len(x_points[0, :])
    w = cp.Variable(numFeatures)
    objective = cp.Minimize(1 / 2 * cp.sum_squares(y_points - cp.matmul(x_points, w))
                            + alpha * cp.norm1(w)) #+ (1-alpha)*cp.sum_squares(w))
    problem = cp.Problem(objective)
    problem.solve(solver=cp.ECOS)

    coef = w.value
    index = np.argsort(-np.abs(coef))  # Sort the coefficient, find the top ones
    cur_basis = []
    print("Found the following sparse low degree polynomial:\n f = ", end="")
    for i in range(0, nMono):
        if i > 0 and coef[index[i]] >= 0:
            print(" +", end="")
        print("%.3f %s" % (coef[index[i]], extendedNames[index[i]]),
              end="")  # Print the top coefficient value, the corresponding feature IDs
        cur_basis.append((coef[index[i]], index[i]))  # Add the basis, and its position, only add top nMono
    print("")
    learnedFeature.append(cur_basis[:])

    # Calculating the best configs to find the best possible configs from the predictor.
    mapped_count = np.zeros(N)  # Initialize the count matrix

    for cur_monomial in cur_basis:  # For every important feature (a monomial) that we learned
        for cur_index in featureIDList[cur_monomial[1]]:  # For every variable in the monomial
            mapped_count[cur_index] += 1  # We update its count

    config_enumerate = np.zeros(
        N)  # Use this array to enumerate all possible configuration (to find the minimum for the current sparse polynomial)
    l = []  # All relevant variables.
    for i in range(0, N):  # We only need to enumerate the value for those relevant variables
        if mapped_count[
            i] > 0:  # This part can be made slightly faster. If count=1, we can always set the best value for this variable.
            l.append(i)

    lists = []
    for i in range(0, 2 ** len(l)):  # for every possible configuration
        for j in range(0, len(l)):
            config_enumerate[l[j]] = 1 if ((i % (2 ** (j + 1))) // (2 ** j) == 0) else -1
        score = 0
        for cur_monomial in cur_basis:
            base = cur_monomial[0]
            for cur_index in featureIDList[cur_monomial[1]]:
                base = base * config_enumerate[cur_index]
            score = score + base
        lists.append((config_enumerate.copy(), score))
    lists.sort(key=lambda x: x[1])
    maskList.append(lists[:t])

    return maskList, learnedFeature
