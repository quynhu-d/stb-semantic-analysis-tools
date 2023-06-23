import numpy as np

def perm_distribution(arr, n, m):
    pr = dict()
    for i in range(len(arr) - n + 1):
        perm = np.argsort(arr[i:i+n,:m], axis=0)    # sort each component, (n!)**m possible permutations
        if perm.tobytes() not in pr:
            pr[perm.tobytes()] = 1
        else:
            pr[perm.tobytes()] += 1
    return np.fromiter(pr.values(), dtype='float64') / (len(arr) - n + 1)

def shannon(pr):
    pr = pr[pr != 0]
    return -(pr * np.log(pr)).sum()

def q_0(n, m):
    N = np.math.factorial(n) ** m
    log_N = m * np.log(np.math.factorial(n))
    return 1 / (np.log(2) + log_N + (N + 1)/(2 * N) * np.log(1 / (N + 1)) - log_N / 2)    # 1 / q([1,0..0],p_e)

def q(pr, n, m):
    N = np.math.factorial(n) ** m
    log_N = m * np.log(np.math.factorial(n))
    q_j = (N - len(pr)) / (2 * N) * (np.log(2) + log_N) + shannon((pr + 1 / N) * 0.5)    # s((p + p_e) / 2)
    q_j -= (shannon(pr) / 2)    # s(p) / 2
    q_j -= (log_N / 2)    # s(p_e) / 2
    q_j *= q_0(n, m)    # normalization
    return q_j

def entropy(pr, n, m):
    log_N = m * np.log(np.math.factorial(n))
    return shannon(pr) / log_N

def complexity(pr, n, m, entropy):
    return q(pr, n, m) * entropy

def entropy_complexity(arr, n, m):
    pr = perm_distribution(arr, n, m)
    ent = entropy(pr, n, m)
    comp = complexity(pr, n, m, ent)    
    return ent, comp
