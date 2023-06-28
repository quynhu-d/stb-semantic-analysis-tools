import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
sns.set_style('whitegrid')

def s_b(N, p, k):
    if p != 0:
        return -(k * p * math.log(p) + (1 - p * k) * math.log((1 - p * k) / (N - k)))
    else:
        return math.log(N - k)

def s_b_1(N, p, k):
    return -(
        k * (p + 1 / N) / 2 * math.log((p + 1 / N) / 2) + \
        (N - k) * ((1 - p * k) / (N - k) + 1 / N) / 2 * math.log((((1 - p * k) / (N - k)) + 1 / N) / 2))

def entropy_b(N, p, k):
    return s_b(N, p, k) / math.log(N)

def q_0_b(N):
    return 1 / (s_b(N, 1 / (2 * N), N - 1) - math.log(N) / 2)

def q_j_b(N, p, k):
    return q_0_b(N) * (s_b_1(N, p, k) - s_b(N, p, k) / 2 - math.log(N) / 2)

def complexity_b(N, p, k, ent=None):
    if ent is None:
        ent = entropy_b(N, p, k)
    return q_j_b(N, p, k) * ent

def entropy_complexity_b(N, p, k):
    e_b = entropy_b(N, p, k)
    c_b = complexity_b(N, p, k, e_b)
    return e_b, c_b

def get_borders(n, m):
    N = np.math.factorial(n)**m
    i = 1
    entropy = []
    complexity = []
    while i * 100 < N:
        for k in range(N - i * 100, N - 1, i):
            e_b, c_b = entropy_complexity_b(N, 0, k)
            entropy.append(e_b)
            complexity.append(c_b)
        i *= 2
    for k in range(0, N - 1, i):
        e_b, c_b = entropy_complexity_b(N, 0, k)
        entropy.append(e_b)
        complexity.append(c_b)
    idx = np.argsort(entropy)
    entropy = np.array(entropy)[idx]
    complexity = np.array(complexity)[idx]
    max_ec = np.vstack([entropy, complexity]).T
    max_ec = np.vstack([[0,0],max_ec,[1,0]])
#     p = 0.01
    entropy = []
    complexity = []
    for p in np.arange(0.01,0.99,0.01):
#         p += 0.01
            e_b, c_b = entropy_complexity_b(N, p, 1)
            entropy.append(e_b)
            complexity.append(c_b)
    idx = np.argsort(entropy)
    entropy = np.array(entropy)[idx]
    complexity = np.array(complexity)[idx]
    min_ec = np.vstack([entropy, complexity]).T
    min_ec = np.vstack([[0,0], min_ec,[1,0]])
    del entropy
    del complexity
    return min_ec, max_ec

def plot_distributions(n, m, table=None, borders=None, lang='vn', ax=None, title='', color_palette=None, n_samples=20):
    if ax is None:
        ax = plt.gca()
    if borders is None:
        min_ec, max_ec = get_borders(n, m)
    else:
        min_ec, max_ec = borders
    
    if table is None:
        raise 'Table is empty'
    else:
        my_table = table[(table['lang'] == lang) & (table['n'] == n) & (table['m'] == m)]
        if 'text_type' in my_table.columns:
            my_table = my_table.groupby('text_type').sample(n=n_samples)
        
    if 'text_type' in my_table.columns:
        if not color_palette:
            color_palette = {
                "lit": (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),    # blue
                "gpt3": (1.0, 0.4980392156862745, 0.054901960784313725),    # orange (see sns.color_palette())
                "bot": (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)    # green
            }
        sns.scatterplot(
            x=my_table['entropy'], y=my_table['complexity'], hue=my_table['text_type'],
            ax=ax, s=20, palette=color_palette, alpha=.7
        )
    else:
        my_table = my_table.sample(n_samples)
        sns.scatterplot(x=my_table['entropy'], y=my_table['complexity'], color='blue', s=5)
    sns.lineplot(x=min_ec[:,0],y=min_ec[:,1],color='r',ax=ax,alpha=.3)
    sns.lineplot(x=max_ec[:,0],y=max_ec[:,1],color='r',ax=ax,alpha=.3)
    ax.set_title(title)
#     plt.rc('xtick', labelsize=10)
#     plt.rc('ytick', labelsize=10)
    return
