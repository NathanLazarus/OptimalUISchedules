# #!/usr/bin/env python3.7

# # Copyright 2021, Gurobi Optimization, LLC

# # This example formulates and solves the following simple MIP model:
# #  maximize
# #        x +   y + 2 z
# #  subject to
# #        x + 2 y + 3 z <= 4
# #        x +   y       >= 1
# #        x, y, z binary

# import gurobipy as gp
# from gurobipy import GRB

# try:

#     # Create a new model
#     m = gp.Model("mip1")

#     # Create variables
#     x = m.addVar(vtype=GRB.BINARY, name="x")
#     y = m.addVar(vtype=GRB.BINARY, name="y")
#     z = m.addVar(vtype=GRB.BINARY, name="z")

#     # Set objective
#     m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)

#     # Add constraint: x + 2 y + 3 z <= 4
#     m.addConstr(x + 2 * y + 3 * z <= 4, "c0")

#     # Add constraint: x + y >= 1
#     m.addConstr(x + y >= 1, "c1")

#     # Optimize model
#     m.optimize()

#     for v in m.getVars():
#         print('%s %g' % (v.varName, v.x))

#     print('Obj: %g' % m.objVal)

# except gp.GurobiError as e:
#     print('Error code ' + str(e.errno) + ': ' + str(e))

# except AttributeError:
#     print('Encountered an attribute error')

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

def utility(state_today, state_tomorrow):
    diffs = state_today.astype('float64') - state_tomorrow
    diffs[diffs < 0] += -np.inf
    diffs[diffs > 1] += -0.5
    return diffs

try:

    # Create a new model
    m = gp.Model("matrix1")

    # Create variables
    x = m.addMVar(shape=3, name="x")

    # Set objective
    obj = np.array([1, 1, 1])
    m.setObjective(obj @ x, GRB.MINIMIZE)


    beta = 0.95

    val = np.array([1 - beta, 1, -beta, 1, -beta, -beta, 1, 1 - beta, 1, -beta, -beta, 1, -beta, 1, 1 - beta])
    row = np.array([0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 8])
    col = np.array([0, 0, 1, 0, 2, 0, 1, 1, 1, 2, 0, 2, 1, 2, 2])

    A = sp.csr_matrix((val, (row, col)), shape=(9, 3))

    print(A.todense())

    # Build rhs vector
    rhs = np.array([0, -np.inf, -np.inf, 1, 0, -np.inf, 1.5, 1, 0])

    # or, equivalently
    # startstate = np.array([0,0,0,1,1,1,2,2,2])
    # endstate = np.array([0,1,2,0,1,2,0,1,2])
    # rhs = utility(startstate, endstate)

    # Add constraints
    m.addConstr(A @ x >= rhs, name="c")

    # Optimize model
    m.optimize()

    print(x.X)
    print('Obj: %g' % m.objVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')