'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3
    
def can_move(model, r, c, r_next, c_next):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()
    r - the row of the agent's current location
    c - the column of the agent's current location
    a - the action the agent is trying to take (0 (left), 1 (up), 2 (right), or 3 (down))

    Output:
    True if the agent can move in the specified direction, False otherwise.
    '''
    if r_next < 0 or r_next >= model.M or c_next < 0 or c_next >= model.N or model.W[r_next, c_next] or model.T[r, c]:
        return False
    else:
        return True


def prob_or_zero(model, r,c, r_next, c_next, prob):
    # return probability of moving to r_next, c_next from r, c given action a
    if (can_move(model, r, c, r_next, c_next)):
        return prob
    else:
        return 0



def compute_transition_matrix(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
    P = np.zeros((model.M, model.N, 4, model.M, model.N))
    for r in range(model.M):
        for c in range(model.N):
            for a in range(4):
                # if the agent is in a terminal state, it stays there
                if model.T[r, c]:
                    P[r, c, a, r, c] = 0 # does not move
                    continue
                else:
                    if a == 0: # left
                        if c-1 < 0: # out of bounds
                            P[r,c,a,r,c] += model.D[r,c,0] # does not move
                        elif model.W[r, c-1]: #     wall
                            P[r,c,a,r,c] += model.D[r,c,0] # does not move
                        else:
                            P[r,c,a,r,c-1] += model.D[r,c,0]
                        if r+1 >= model.M or model.W[r+1, c]:
                            P[r,c,a,r,c]+= model.D[r,c,1] # does not move
                        else:
                            P[r,c,a,r+1,c] += model.D[r,c,1] # down
                        if r-1 < 0 or model.W[r-1, c]: # up
                            P[r,c,a,r,c] += model.D[r,c,2] # does not move
                        else:
                            P[r,c,a,r-1, c] += model.D[r,c,2] # does not move
                    elif a == 1: # up
                        if r-1 < 0:
                            P[r,c,a,r,c] += model.D[r,c,0]
                        elif model.W[r-1, c]:
                            P[r,c,a,r,c] += model.D[r,c,0] # does not move
                        else:
                            P[r,c,a,r-1,c] += model.D[r,c,0] # up
                        if c-1 < 0 or model.W[r, c-1]:
                            P[r,c,a,r,c] += model.D[r,c,1]
                        else:
                            P[r,c,a,r,c-1] += model.D[r,c,1] # left
                        if c+1 >= model.N or model.W[r, c+1]:
                            P[r,c,a,r,c] += model.D[r,c,2]
                        else:
                            P[r,c,a,r,c+1] += model.D[r,c,2] # right
                    elif a == 2: # right
                        if c+1 >= model.N:
                            P[r,c,a,r,c] += model.D[r,c,0]
                        elif model.W[r, c+1]:
                            P[r,c,a,r,c] += model.D[r,c,0]
                        else:
                            P[r,c,a,r,c+1] += model.D[r,c,0] # right
                        if r-1 < 0 or model.W[r-1, c]:
                            P[r,c,a,r,c] += model.D[r,c,1]
                        else:
                            P[r,c,a,r-1,c] += model.D[r,c,1] # up
                        if r+1 >= model.M or model.W[r+1, c]:
                            P[r,c,a,r,c] += model.D[r,c,2]
                        else:
                            P[r,c,a,r+1,c] += model.D[r,c,2] # down
                    elif a == 3: # down
                        if r+1 >= model.M:
                            P[r,c,a,r,c] += model.D[r,c,0]
                        elif model.W[r+1, c]:
                            P[r,c,a,r,c] += model.D[r,c,0]
                        else:
                            P[r,c,a,r+1,c] += model.D[r,c,0] # down
                        if c+1 >= model.N or model.W[r, c+1]:
                            P[r,c,a,r,c] += model.D[r,c,1]
                        else:
                            P[r,c,a,r,c+1] += model.D[r,c,1] # right
                        if c-1 < 0 or model.W[r, c-1]:
                            P[r,c,a,r,c] += model.D[r,c,2]
                        else:
                            P[r,c,a,r,c-1] += model.D[r,c,2] # left


    return P

def best_a(P, U_current, r, c, model):
    max_dot_product = 0
    for a in range(4):
        # dot product of the flattened P and U_current(next state)
        # dot_prod = P[r, c, a, r, c] * U_current[r, c] # does not move
        #  That is, we can rewrite the update rule as some matrix operations and then use numpy's builtin functions to compute them. For example, the summation in the equation is actually an inner product of P and Ui
        dot_prod = np.sum(P[r, c, a, :, :] * U_current)
        # if a == 0: # left
        #     # calculating the sum of the utilities at all the possible states the agent can end up in after taking action a from state s.
        #     # top, left, bottom
        #     if (can_move(model, r, c, r, c-1)):
        #         dot_prod += P[r, c, a, r, c-1] * U_current[r, c-1] # left
        #     if (can_move(model, r, c, r+1, c)):
        #         dot_prod += P[r, c, a, r+1, c] * U_current[r+1, c] # down
        #     if (can_move(model, r, c, r-1, c)):
        #         dot_prod += P[r, c, a, r-1, c] * U_current[r-1, c] # up
        # elif a == 1: # up
        #     if (can_move(model, r, c, r-1, c)):
        #         dot_prod += P[r, c, a, r-1, c] * U_current[r-1, c] # up
        #     if (can_move(model, r, c, r, c-1)):
        #         dot_prod += P[r, c, a, r, c-1] * U_current[r, c-1] # left
        #     if (can_move(model, r, c, r, c+1)):
        #         dot_prod += P[r, c, a, r, c+1] * U_current[r, c+1] # right
        # elif a == 2: # right
        #     if (can_move(model, r, c, r, c+1)):
        #         dot_prod += P[r, c, a, r, c+1] * U_current[r, c+1]
        #     if (can_move(model, r, c, r-1, c)):
        #         dot_prod += P[r, c, a, r-1, c] * U_current[r-1, c]
        #     if (can_move(model, r, c, r+1, c)):
        #         dot_prod += P[r, c, a, r+1, c] * U_current[r+1, c]
        # elif a == 3: #  down
        #     if (can_move(model, r, c, r+1, c)):
        #         dot_prod += P[r, c, a, r+1, c] * U_current[r+1, c]
        #     if (can_move(model, r, c, r, c+1)):
        #         dot_prod += P[r, c, a, r, c+1] * U_current[r, c+1]
        #     if (can_move(model, r, c, r, c-1)):
        #         dot_prod += P[r, c, a, r, c-1] * U_current[r, c-1]
        if dot_prod > max_dot_product:
            max_dot_product = dot_prod
    return max_dot_product

def update_utility(model, P, U_current):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    P - The precomputed transition matrix returned by compute_transition_matrix()
    U_current - The current utility function, which is an M x N array

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    U_next = np.zeros((model.M, model.N))
    for r in range(model.M):
        for c in range(model.N):
            U_next[r, c] = model.R[r, c] + model.gamma * np.max(np.sum(P[r, c] * U_current, axis=(1,2)))
    return U_next

def value_iteration(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    P = compute_transition_matrix(model)
    U_current = np.zeros((model.M, model.N))
    U_next = update_utility(model, P, U_current)
    count = 0
    max_iter = 100 
    while (np.max(np.abs(U_next - U_current)) >= epsilon) and count < max_iter:
        U_current = U_next
        U_next = update_utility(model, P, U_current)
        count += 1
    return U_next

if __name__ == "__main__":
    import utils
    model = utils.load_MDP('models/small.json')
    model.visualize()
    U = value_iteration(model)
    model.visualize(U)
