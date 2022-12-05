import gurobipy as gp
import numpy as np
import scipy.sparse as sp

numks = 5
kmin = 0
kmax = 1
gamma = -10
beta = 0.9


def output(k):
    return 2 * k - (k ** 2) / 2


def consumption(state_today, state_tomorrow):
    return output(state_today) - state_tomorrow


def utility(state_today, state_tomorrow, gamma):
    return -np.exp(gamma * consumption(state_today, state_tomorrow))


kvec = np.linspace(kmin + kmax / numks, kmax, numks)
# kvec = kmin + np.arange(1, numks + 1, dtype="float64") / numks

startstate = np.repeat(kvec, numks)
startstate_inds = np.repeat(np.arange(numks), numks)

endstate = np.tile(kvec, numks)
endstate_inds = np.tile(np.arange(numks), numks)

utilities = utility(startstate, endstate, gamma)

rows = np.repeat(np.arange(numks ** 2), 2)
cols = np.ravel([startstate_inds, endstate_inds], "F")
val = np.tile(np.array([1, -beta]), numks ** 2)

valueMat = sp.csr_matrix((val, (rows, cols)), shape = (numks ** 2, numks))

# print(valueMat.todense())

try:

    # Create a new model
    m = gp.Model("growth")

    # Create variables
    x = m.addMVar(shape = numks, name = "x", lb = -gp.GRB.INFINITY)

    # Set objective
    obj = np.ones(numks)
    m.setObjective(obj @ x, gp.GRB.MINIMIZE)

    # Add constraints
    m.addConstr(valueMat @ x >= utilities, name = "constraint")

    # Use the dual simplex algorithm only (the default is to run 
    # the barrier as well in parallel)
    m.setParam("Method", 1)
    m.setParam("Presolve", 1)

    # Optimize model
    m.optimize()

    print(x.X)
    print("Obj: %g" % m.objVal)

except gp.GurobiError as e:
    print("Error code " + str(e.errno) + ": " + str(e))

except AttributeError:
    print("Encountered an attribute error")


# Parameter Tuning:

# m.tune()

# if m.tuneResultCount > 0:

#     # Load the best tuned parameters into the model
#     m.getTuneResult(0)

#     # Write tuned parameters to a file
#     m.write('tune.prm')