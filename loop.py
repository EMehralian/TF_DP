import sys
import numpy as np
import math
import matplotlib.pyplot as plt


# sigmas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# sigmas = [0.2]
# for sigma in sigmas:
#     for seed in range(10):
#         print(sigma, seed)
#         sys.argv = ['x', str(seed), str(sigma)]
#         exec(open('dpql.py').read())

# 'dpql_s_ndp.txt',
# 'dpql_t_ndp.txt',

# , ('dpql_t_3.txt', "eps = 0.3")
# , ('dpql_t_4.txt', "eps = 0.4")
# ('data/dpql_t_2_prime.txt', " eps = 0.2, goal 0.5 => 0.6"),
# ,  ('data/dpql_t_0.txt', "no noise, goal 0.5 => 0.6")
# ('dpql_t_2_prime_prime.txt', " eps = 0.2 =>0, goal 0.5 => 0.6"),

file_names = [('dpql_ep2_s.txt', " eps = 0.2,  goal 0.5 => 0.8 from scratch"),('dpql_ep2_t8.txt', " eps = 0.2 goal 0.5 => 0.6")]#,  ('dpql_ep2_t.txt', " eps = 0.2"),('dpql_ep4_t.txt', " eps = 0.4")]# =>0, goal 0.5 => 0.6")]#, ('dpql_t_0.txt', "no noise, goal 0.5 => 0.6")]

super_data = []
for file_name, label in file_names:

    data = []
    with open(file_name) as f:
        content = f.readlines()
    for x in content:
        data.append(np.array(x.split(',')[:-1]).astype(np.float))
    data = np.array(data)
    super_data.append((data, label))


for data, label in super_data:
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) / math.sqrt(len(data))
    plt.plot(mean, label=label)
    plt.fill_between(range(len(data[0])), mean - std, mean + std, alpha=0.3)

plt.xlabel("Number of samples trained")
plt.ylabel("Score")
# plt.title(title)
plt.legend()
plt.show()

