import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import json

# Set random seed
np.random.seed(129)

# define function to initialize network
def init_network(num_bank, max_debt, max_cds, max_weight):
    in_debt = [[] for _ in range(num_bank)]
    out_debt = [([], []) for _ in range(num_bank)] # (bank ids, weight list)
    out_cds = [([], [], []) for _ in range(num_bank)] # (bank ids, reference bank ids, weight list)
    inf_cds = [[] for _ in range(num_bank)]
    candidates = list(range(num_bank))

    for i in range(num_bank):
        for bank in np.random.choice(list(range(num_bank)), np.random.randint(2, max_debt + 1), replace = False):
            if (bank!=i) and (bank not in in_debt[i]) and (bank not in out_debt[i][0]):
                out_debt[i][0].append(bank)
                out_debt[i][1].append(np.random.randint(1, max_weight + 1))
                in_debt[bank].append(i)

        for bank in np.random.choice(list(range(num_bank)), np.random.randint(0, max_cds + 1), replace = False):
            if (bank!=i) and (i not in out_cds[bank][0]):
                out_cds[i][0].append(bank)
                out_cds[i][2].append(np.random.randint(1, max_weight + 1))
                tmp = np.random.randint(0, num_bank)
                while (tmp==i or tmp==bank):
                    tmp = np.random.randint(0, num_bank)

                out_cds[i][1].append(tmp)
                inf_cds[tmp].append((i, len(out_cds[i][0]) - 1))
                
    return (out_debt, out_cds, inf_cds)

# Implement R(a, l)
def Recovery(a, l):
    r = np.ones(a.shape)
    for i in range(a.shape[0]):
        if a[i]<l[i]:
            r[i] = a[i]/l[i]
    
    return r

# define experiments function
def experiment(num_bank, out_debt, out_cds, inf_cds, max_steps = 20000, eps = 1e-6):
    assets = np.zeros(num_bank)
    liabilities = np.zeros(num_bank)
    recovery = np.ones(num_bank)
    
    # initialize assets and liabilities
    for i in range(num_bank):
        for j in range(len(out_debt[i][0])):
            liabilities[i] += out_debt[i][1][j]
            assets[out_debt[i][0][j]] += out_debt[i][1][j]

    new_r = Recovery(assets, liabilities)
    r_change = np.argwhere((np.abs(new_r - recovery)/recovery)>eps).squeeze(axis = 1)
    
    result = -1
    
    for i in range(max_steps):
        # select a bank to update recovery rate
        u = np.random.choice(r_change, 1)[0]
        pre_r = recovery[u]
        recovery[u] = new_r[u]
        for j in range(len(out_debt[u][0])):
            assets[out_debt[u][0][j]] += out_debt[u][1][j]*(recovery[u] - pre_r)

        for j in range(len(out_cds[u][0])):
            assets[out_cds[u][0][j]] += out_cds[u][2][j]*(recovery[u] - pre_r)*(1 - recovery[out_cds[u][1][j]])

        for cds_bank, k in inf_cds[u]:
            liabilities[cds_bank] += out_cds[cds_bank][2][k]*(pre_r - recovery[u])*recovery[cds_bank]
            assets[out_cds[cds_bank][0][k]] += out_cds[cds_bank][2][k]*(pre_r - recovery[u])*recovery[cds_bank]

        new_r = Recovery(assets, liabilities)
        r_change = np.argwhere((np.abs(new_r - recovery)/(recovery + eps))>eps).squeeze(axis = 1)

        if len(r_change)==0:
            result = i
            break
            
    return result


# Experiments with Fixed Parameters
print("Run simulations with fixed parameters")

# define hyperparameter
num_banks = [10, 11, 12, 14, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200]
num_simulations = 30
max_weight = 4

# define fixed parameters
max_debt = 8
max_cds = 7

results_fix = []
no_converge_fix = np.zeros(len(num_banks))

for i in tqdm(range(len(num_banks)), desc = "banks number"):
    num_bank = num_banks[i]
    tmp_results = []
    
    for j in tqdm(range(num_simulations), desc = "simulation count", leave = False):
        out_debt, out_cds, inf_cds = init_network(num_bank, max_debt, max_cds, max_weight)
        tmp = experiment(num_bank, out_debt, out_cds, inf_cds)
        if tmp != -1:
            tmp_results.append(tmp)
        else:
            no_converge_fix[i] += 1
            
    results_fix.append(tmp_results)

# show simulation results with fixed data

print("The number of cases without convergence for simulations with fixed parameters is ", no_converge_fix)

mean_steps_fix = np.zeros(len(num_banks))
for i in range(len(results_fix)):
    mean_steps_fix[i] = np.array(results_fix[i]).mean()
    

# Experiments with Non-Fixed Parameters  
print("Run simulations with non-fixed parameters")

results_nonfix = []
no_converge_nonfix = np.zeros(len(num_banks))

for i in tqdm(range(len(num_banks)), desc = "banks number"):
    num_bank = num_banks[i]
    tmp_results = []
    max_debt = int(2.4*np.sqrt(num_bank))
    max_cds = int(2.2*np.sqrt(num_bank))
    
    for j in tqdm(range(num_simulations), desc = "simulation count", leave = False):
        out_debt, out_cds, inf_cds = init_network(num_bank, max_debt, max_cds, max_weight)
        tmp = experiment(num_bank, out_debt, out_cds, inf_cds)
        if tmp != -1:
            tmp_results.append(tmp)
        else:
            no_converge_nonfix[i] += 1
            
    results_nonfix.append(tmp_results)

print("The number of cases without convergence for simulations with non-fixed parameters is ", no_converge_nonfix)

mean_steps_nonfix = np.zeros(len(num_banks))
for i in range(len(results_nonfix)):
    mean_steps_nonfix[i] = np.array(results_nonfix[i]).mean()

result_path = './results'

if not os.path.exists(result_path):
    os.mkdir(result_path)

with open(result_path + '/simulation_results.json', 'w') as f:
    json.dump({'num_banks': num_banks, 'fix_result': results_fix, 'nonfix_result': results_nonfix}, f)

plt.figure(figsize=(8, 5))
plt.plot(num_banks, mean_steps_nonfix, '-o', label = 'Not Fixed Parameters')
plt.plot(num_banks, mean_steps_fix, '-*', label = 'Fixed Parameters')
plt.xlabel("The number of banks in the system", fontsize =12)
plt.ylabel("Average number of steps to converge", fontsize =12)
plt.legend(fontsize =12, loc = 'upper left')
plt.savefig(result_path + '/result.png')
plt.savefig(result_path + '/result.eps')
plt.show()