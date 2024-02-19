import random
import math
import itertools

#UTILS
# ANSI color and style codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[31m"
BLUE = "\033[34m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
RESET = "\033[0m"


#SHARED GLOBAL VARIABLES
p = 524287  # Mersenne prime for modulo arithmetic
state = {'prover': 0, 'verifier': 0}  # sum-check protocol steps
initial_message = 0 # initial message
terminating_message = -1 # terminating message


#PROVER GLOBAL VARIABLES
A = [
    [0, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 1, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 0]
    ] # graph complete
#A = [[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]]  #square
#A = [[0,1,0,0],[1,0,1,1],[0,1,0,0],[0,0,0,0]]  #claw
#A = [[0,1,1,0],[1,0,1,1],[1,1,0,0],[0,1,0,0]]  #pan
#A = [[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]] #spindle
#A = [[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]] # complete

size = len(A) # size of the graph

k = int(math.log2(size)) 
M = 3*k # number of steps
G = 0 # 6 times the number of triangles 
R = [] # list of random generated number

#VERIFIER GLOBAL VARIABLES
check_values = [] # list of list for [g_j(0), g_j(1), g_j(2)] prover sents to verifier


# PROVER 
def prover(message):
    global state, M, initial_message
    # check initial message
    if message == initial_message:
        state['prover'] = 1
    response_message = PStep(state['prover'], M, message)

    # print the step
    if len(response_message) == 4:
        print(f"Step {state['prover']} {BLUE}Prover{RESET}: sends message H = {BOLD}{response_message[3]}{RESET}.")
    print(f"Step {state['prover']} {BLUE}Prover{RESET}: sends message [g_{state['prover']}(0), g_{state['prover']}(1), g_{state['prover']}(2)] = {BOLD}{response_message[0:3]}{RESET}")

    # update state
    state['prover'] += 1

    return response_message

def PStep(n, M, message):
    global R, A, G 
    # first iteration sends both H and evaluations at {0,1,2}
    if n == 1:
        # computes the number of trianlges
        G = G(A)
        print(f"Step {n} {BLUE}Prover{RESET}: computes the number of triangles: {BOLD}{G}{RESET} / 6 = {G // 6}")
        
        # reset random variable list
        R = []
        # Value
        response_message = h(A,R)
        response_message.append(G)
    else:
        # remember the random variable verifier sends
        R.append(message)
        response_message = h(A,R)

    return response_message

# VERIFIER
def verifier(message):
    global state, M, initial_message, terminating_message
    # check initial message
    if message == initial_message:
        state['verifier'] = 1
        response_message = message
        # print initial message
        print(f"Step {state['verifier']} {YELLOW}Verifier{RESET}: sends initial message {UNDERLINE}{BOLD}{response_message}{RESET}{RESET} (ignored by C).")
    else:
        response_message = VStep(state['verifier'], M, message)
        # check terminating message
        if response_message == terminating_message:
            # print the terminating step
            print(f"Step {state['verifier']} {YELLOW}Verifier{RESET}: sends terminating message {UNDERLINE}{BOLD}{response_message}{RESET}{RESET}. \n")
            return response_message
        else:
            # print the verifier step
            print(f"Step {state['verifier']} {YELLOW}Verifier{RESET}: sends message random number {BOLD}{response_message}{RESET}. \n")
            state['verifier'] += 1

    return response_message

def VStep(n, M, message):
    global p, size, check_values, terminating_message
    # verifier Compute the number of triangles G
    if n == 1:
        # reset check values memory
        check_values = []
        # the first 3 element are g_j(0), g_j(1), and g_j(2)
        check_values.append(message[0:3])
        # h1(0) + h1(1) == H?
        if (message[0] + message[1]) % p == G % p:
            print(f"Step {n} {YELLOW}Verifier{RESET}: checks h_{n}(0) + h_{n}(1) == H {GREEN}{message[0]} + {message[1]} == {G % p}{RESET}")
            response_message = random.randint(size,p)
        else:
            print(f"Step {n} {YELLOW}Verifier{RESET}: checks h_{n}(0) + h_{n}(1) == h_{n-1}(r_{n}) {RED}{(message[0] + message[1]) % p} != {g_j(message_prev[0], message_prev[1],message_prev[2], R[-1])}{RESET}")
            response_message = terminating_message
    elif n == M:
        message_prev = check_values[-1]
        # h_{m}(0) + h_{m}(1) ?= h_{m-1}(r_{m-1})
        if (message[0] + message[1]) % p == g_j(message_prev[0], message_prev[1],message_prev[2], R[-1]):
            print(f"Step {n} {YELLOW}Verifier{RESET}: checks h_{n}(0) + h_{n}(1) == h_{n-1}(r_{n-1})   {GREEN}{(message[0] + message[1]) % p} == {g_j(message_prev[0], message_prev[1],message_prev[2], R[-1])}{RESET}")
            response_message = terminating_message
        else:
            print(f"Step {n} {YELLOW}Verifier{RESET}: checks h_{n}(0) + h_{n}(1) == h_{n-1}(r_{n-1}) {RED}{(message[0] + message[1]) % p} != {g_j(message_prev[0], message_prev[1],message_prev[2], R[-1])}{RESET}")
            response_message = terminating_message
    else:
        message_prev = check_values[-1]
        check_values.append(message[0:3])
        # h_{m}(0) + h_{m}(1) ?= h_{m-1}(r_{m-1})
        if (message[0] + message[1]) % p == g_j(message_prev[0], message_prev[1],message_prev[2], R[-1]):
            print(f"Step {n} {YELLOW}Verifier{RESET}: checks h_{n}(0) + h_{n}(1) == h_{n-1}(r_{n-1}) {GREEN}{(message[0] + message[1]) % p} == {g_j(message_prev[0], message_prev[1],message_prev[2], R[-1])}{RESET}")
            response_message = random.randint(0,p)
        else:
            print(f"Step {n} {YELLOW}Verifier{RESET}: checks h_{n}(0) + h_{n}(1) == h_{n-1}(r_{n-1}) {RED}{(message[0] + message[1]) % p} != {g_j(message_prev[0], message_prev[1],message_prev[2], R[-1])}{RESET}")
            response_message = terminating_message
    return response_message

# COUNT TRIANGLE FUNCTIONS
# multilinear Lagrange basis polynomials with interpolating set {0,1}^n
def d(W,X):
  res = 1
  for i in range(len(W)):
    if W[i] == 0:
      res = (res * (1-X[i])) % p
    else:
      res = (res * X[i]) % p
  return res % p

# f(W) = 1 only if there is an edge
def f(A, W):
    global k
    a = 0
    b = 0
    for i in range(k):
        a += W[i] * (2**i)
    for i in range(k):
        b += W[i + k] * (2**i)
    return A[a][b]

# lagrange interpolation 
def F(A, X):
    global p
    res = 0
    for W in itertools.product([0,1], repeat = 2*k):
        res = (res + f(A,W) * d(W,X)) % p
    return res % p

# counting triangles G = 6 * num_of_triangles
def G(A):
    res = 0
    for X_tuple in itertools.product([0,1], repeat = 3*k):
        X = list(X_tuple)
        X1 = X[0:k]
        X2 = X[k:2*k]
        X3 = X[2*k:]
        res += F(A,X1 + X2) * F(A,X1 + X3) * F(A, X2 + X3)
    return res

# h_j(x_j) =ct= g_j(x_j)
def h(A,R):
    global p
    response_vector = []
    
    # compute evaluations at {0,1,2}
    for i in range(3):
        res = 0
        # the first len(R) items are previous random numbers
        for X_tuple in itertools.product([0,1], repeat = 3*k - len(R) - 1):
            # convert tuple to list to allow modifications
            # random number from verifier insert at the front
            X = R + [i] + list(X_tuple)
            # assign the vectors for triangle counting polynomial
            X1 = X[0:k]
            X2 = X[k:2*k]
            X3 = X[2*k:]
            res = (res + F(A,X1 + X2) * F(A,X1 + X3) * F(A, X2 + X3)) % p
        response_vector.append(res % p)
    return response_vector

# g_j(r_j) with langrange interpolation
def g_j(g0, g1, g2, r):
    # apply modulo at every step to avoid rounding error
    term1 = ((g0 * (r - 1) % p) * (r - 2) % p) % p
    term2 = ((g1 * r % p) * (r - 2) % p) % p
    term3 = ((g2 * r % p) * (r - 1) % p) % p
    # p is an odd prime, division by 2 can be handled via multiplication by the modular inverse of 2 mod p
    inverse_of_2 = pow(2, p-2, p)
    result = (term1 * inverse_of_2 - term2 + term3 * inverse_of_2) % p
    return result



# Step 1: Initialize prerequisites before starting the protocol
## All glocal variables are initialize at the beginning

# set the seed
random.seed(420) # I choose the best number in the world :)

# Step 2: Initialize the verifier with the initial message
print("\033[92m" + "===================================================================")
print("        Sum-Check Protocol Started for Counting Triangles!            ")
print("===================================================================" + "\033[0m \n")
verifier(initial_message)

# Step 3: Start the loop for the protocol interaction between prover and verifier
message = initial_message
while True:
    # Step 3.1: Prover generates a new message based on the current message
    message = prover(message)

    # Step 3.2: Verifier processes the new message and potentially updates it
    message = verifier(message)

    # Step 3.3: Check if the message from the verifier indicates termination
    if message == terminating_message:
        break  # Exit the loop if termination condition is met

# Step 4: Finalize and clean up after the protocol has conclude
print("\033[92m" + "===================================================================")
print("                   Sum-Check Protocol Finished!                      ")
print("===================================================================" + "\033[0m")



