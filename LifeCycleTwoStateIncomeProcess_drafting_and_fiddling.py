import gurobipy as gp
import numpy as np
import scipy.sparse as sp
import sys

numks = 5
amin = 0
amax = 1
gamma = -10
beta = 0.9
N_income_states = 2
pstay = 0.75


def income(k):
    return 2 * k - (k ** 2) / 2


def consumption(state_today, state_tomorrow):
    return income(state_today) - state_tomorrow


def utility(state_today, state_tomorrow, gamma):
    return -np.exp(gamma * consumption(state_today, state_tomorrow))

def u_of_c(consumption, gamma):
    return -np.exp(gamma * consumption)

def get_probs(kvec, k_grid_space, kval, cval):
    probs = np.zeros([len(kvec), 2])
    k_remaining = kval - cval
    for k_ind, this_k in enumerate(kvec):
        probs[k_ind, 0] = k_ind
        if abs(this_k - k_remaining) < k_grid_space:
            probs[k_ind, 1] = (k_grid_space - abs(this_k - k_remaining)) / k_grid_space

    return probs


def get_probs2(kvec, k_grid_space, remaining_k, current_income_state, pstay):
    probs = np.zeros([len(kvec), 2])
    for state_ind, this_state in enumerate(kvec):
        probs[state_ind, 0] = state_ind
        this_income_state = this_state[0]
        this_capital_state = float(this_state[1:])
        if current_income_state == this_income_state:
            if abs(this_capital_state - remaining_k) < k_grid_space:
                probs[state_ind, 1] = (k_grid_space - abs(this_capital_state - remaining_k)) / k_grid_space * pstay
        else:
            if abs(this_capital_state - remaining_k) < k_grid_space:
                probs[state_ind, 1] = (k_grid_space - abs(this_capital_state - remaining_k)) / k_grid_space * (1 - pstay)
    return probs

# startstate = np.repeat(kvec, numks)
# startstate_inds = np.repeat(np.arange(numks), numks)

# endstate = np.tile(kvec, numks)
# endstate_inds = np.tile(np.arange(numks), numks)

# print(startstate)
# print(endstate)

# num_c_choices = 5
# consumptiongrid = np.linspace(0, 0.8, num_c_choices)
# valueMat = sp.csr_matrix((numks ** 2, numks))
# counter = 0
# utilities = np.zeros(numks ** 2)
# for cind, c in enumerate(consumptiongrid):
#     for kind, k in enumerate(kvec):
#         if k >= c + 0.2:
#             utilities[counter] = u_of_c(c, gamma)
#             valueMat[counter, kind] += 1
#             valueMat[counter, kind - cind] += -beta * 0.5
#             valueMat[counter, min(kind - cind + 1, len(kvec) - 1)] += -beta * 0.5
#             counter += 1

# valueMat = valueMat[:counter,:]
# utilities = utilities[:counter]
# print(utilities)
# print(valueMat.todense())



num_c_choices = 21
consumptiongrid = np.linspace(0, 1, num_c_choices)
valueMat = sp.csr_matrix((numks * num_c_choices, numks * N_income_states))
counter = 0
utilities = np.zeros(numks * num_c_choices)

# twostate_kvec = zip(np.tile(kvec, 2).tostring(), ['h'] * numks + ['l'] * numks)
twostate_kvec = [''.join(x) for x in zip(['h'] * numks + ['l'] * numks, ['0.2', '0.4', '0.6', '0.8', '1'] * 2)]
print(twostate_kvec)

for cind, c in enumerate(consumptiongrid):
    for kind, state in enumerate(twostate_kvec):
        if state[0] == 'h':
            income = 0.2
        elif state[0] == 'l':
            income = 0

        capital = float(state[1:])
        remaining_k = capital + income - c
        print('c', c)
        print('state', state)
        print('remaining_k', remaining_k)
        print(remaining_k >= (0.2 - 1e-9) and remaining_k <= (1 + 1e-9))
        if remaining_k >= 0.2 and remaining_k <= 1:
            utilities[counter] = u_of_c(c, gamma)
            valueMat[counter, kind] += 1
            for transition in get_probs2(twostate_kvec, 0.2, remaining_k, state[0], pstay):
                if transition[1] > 1e-10:
                    valueMat[counter, int(transition[0])] += -beta * transition[1]
            # print(valueMat[counter,:].todense())
            counter += 1

valueMat = valueMat[:counter,:]
utilities = utilities[:counter]
print(utilities)

print('**************')
print(valueMat.todense())


# utilities = utility(startstate, endstate, gamma)


# print(income(kvec))
# print(consumptiongrid)
# print(np.tile(kvec, num_c_choices) - np.tile(consumptiongrid, numks))
# print(consumption(startstate, endstate))
# print(consumption(startstate, endstate) > 0)


# rows = np.repeat(np.arange(numks ** 2), 2)
# cols = np.ravel([startstate_inds, endstate_inds], "F")
# val = np.tile(np.array([1, -beta]), numks ** 2)

# valueMat = sp.csr_matrix((val, (rows, cols)), shape = (numks ** 2, numks))

# print(valueMat.todense())

print(utilities)
print(utilities.shape)
print(valueMat.shape)
print(numks)

try:

    # Create a new model
    m = gp.Model("growth")

    # Create variables
    x = m.addMVar(shape = numks * N_income_states, name = "x", lb = -gp.GRB.INFINITY)

    # Set objective
    obj = np.ones(numks * N_income_states)
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