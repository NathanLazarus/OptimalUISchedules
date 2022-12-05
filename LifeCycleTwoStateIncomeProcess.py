import gurobipy as gp
import numpy as np
import scipy.sparse as sp
import sys

N_asset_states = 5
N_income_states = 2
num_c_choices = 21

Nstates = N_asset_states * N_income_states

income_state_transition_matrix = np.array([[0.75, 0.25], [0.25, 0.75]])

amin = 0
amax = 1
gamma = -10
beta = 0.9


# def income(k):
#     return 2 * k - (k ** 2) / 2


# def consumption(state_today, state_tomorrow):
#     return income(state_today) - state_tomorrow


# def utility(state_today, state_tomorrow, gamma):
#     return -np.exp(gamma * consumption(state_today, state_tomorrow))

def u_of_c(consumption, gamma):
    return -np.exp(gamma * consumption)


def p_state_next_period(Nstates, statevec, assetvec, k_grid_space, remaining_k, current_state, income_state_transition_matrix):
    probs = np.zeros(Nstates)
    current_income_state = current_state[0]
    for state_ind, this_state in enumerate(statevec):
        this_income_state, this_asset_state = this_state
        this_assets = assetvec[this_asset_state]
        if abs(this_assets - remaining_k) < k_grid_space:
            probs[state_ind] = (k_grid_space - abs(this_assets - remaining_k)) / k_grid_space * income_state_transition_matrix[int(current_income_state), int(this_income_state)]
    future_states = np.arange(Nstates)[probs > 1e-13]
    return future_states, probs[probs > 1e-13]

asset_states = np.arange(N_asset_states)
assetvec = np.linspace(amin + amax / N_asset_states, amax, N_asset_states)

high_income = 0.2
low_income = 0

income_states = np.arange(N_income_states)
incomevec = np.array([high_income, low_income])


all_states = np.vstack([np.repeat(income_states, N_asset_states), np.tile(asset_states, N_income_states)]).T

consumptiongrid = np.linspace(0, max(assetvec) + max(incomevec) - min(assetvec), num_c_choices)

valueMat = sp.csr_matrix((Nstates * num_c_choices, Nstates))
utilities = np.zeros(Nstates * num_c_choices)

test_dense_valueMat = np.zeros((Nstates * num_c_choices, Nstates))

counter = 0

# nonzero_entries_in_valueMat = np.zeros((Nstates * num_c_choices, Nstates))

for c in consumptiongrid:
    for current_state_ind, state in enumerate(all_states):
        current_income_state, current_asset_state = state
        current_assets = assetvec[current_asset_state]
        current_income = incomevec[current_income_state]
        remaining_assets = current_assets + current_income - c
        is_valid_consumption_asset_combo = remaining_assets >= (min(assetvec) - 1e-9) and remaining_assets <= (max(assetvec) + 1e-9)
        if is_valid_consumption_asset_combo:
            utilities[counter] = u_of_c(c, gamma)
            valueMat[counter, current_state_ind] += 1
            test_dense_valueMat[counter, current_state_ind] += 1
            future_states, p_transition_to_future_state = p_state_next_period(Nstates, all_states, assetvec, 0.2, remaining_assets, state, income_state_transition_matrix)
            test_dense_valueMat[counter, future_states] += -beta * p_transition_to_future_state

            counter += 1

# valueMat = valueMat[:counter,:]
valueMat = sp.csr_matrix(test_dense_valueMat[:counter,:])
utilities = utilities[:counter]

print(valueMat.todense())
print(utilities)

try:

    # Create a new model
    m = gp.Model("growth")

    # Create variables
    x = m.addMVar(shape = Nstates, name = "x", lb = -gp.GRB.INFINITY)

    # Set objective
    obj = np.ones(Nstates)
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