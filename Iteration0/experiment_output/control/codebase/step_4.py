# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import itertools

if __name__ == '__main__':
    var_params_path = 'data/var_model_parameters.npz'
    var_data = np.load(var_params_path, allow_pickle=True)
    A_mat = var_data['A']
    Sigma = var_data['Sigma']
    p = int(var_data['p'])
    data_cols = var_data['data_cols']
    K = Sigma.shape[0]
    B = []
    for j in range(1, p + 1):
        B.append(A_mat[:, 1 + (j-1)*K : 1 + j*K])
    perms_and_signs = []
    for perm in itertools.permutations([0, 1, 2]):
        for signs in itertools.product([1, -1], repeat=3):
            perms_and_signs.append((list(perm), np.array(signs)))
    def check_sign_restrictions_fast(A0):
        for perm, signs in perms_and_signs:
            A = A0[:, perm] * signs
            if A[0, 0] > 0 and A[1, 0] < 0 and A[2, 0] < 0:
                if A[1, 1] > 0 and A[2, 1] > 0:
                    if A[1, 2] > 0 and A[2, 2] < 0:
                        return True, A
        return False, None
    np.random.seed(42)
    P = np.linalg.cholesky(Sigma)
    strict_exogeneity_possible = (P[0, 0] > 0 and P[1, 0] < 0 and P[2, 0] < 0)
    if strict_exogeneity_possible:
        print('Data supports strict contemporaneous exogeneity for Mortgage Rates. Using block-diagonal Q.')
    else:
        print('Data does NOT support strict contemporaneous exogeneity with the required sign restrictions.')
        print('Falling back to full 3x3 Rubio-Ramirez algorithm to satisfy sign restrictions.')
    accepted_A0 = []
    attempts = 0
    max_accepted = 1000
    max_attempts = 5000000
    print('Starting Rubio-Ramirez et al. (2010) algorithm for sign restrictions...')
    start_time = time.time()
    while len(accepted_A0) < max_accepted and attempts < max_attempts:
        attempts += 1
        if attempts == 100000 and len(accepted_A0) == 0 and strict_exogeneity_possible:
            print('Warning: Block-diagonal Q failed to find valid matrices. Falling back to full 3x3 Q.')
            strict_exogeneity_possible = False
        if strict_exogeneity_possible:
            Q = np.zeros((3, 3))
            Q[0, 0] = 1.0
            X = np.random.randn(2, 2)
            Q_sub, R_sub = np.linalg.qr(X)
            for i in range(2):
                if R_sub[i, i] < 0:
                    Q_sub[:, i] = -Q_sub[:, i]
            Q[1:, 1:] = Q_sub
        else:
            X = np.random.randn(3, 3)
            Q, R = np.linalg.qr(X)
            for i in range(3):
                if R[i, i] < 0:
                    Q[:, i] = -Q[:, i]
        A0_candidate = P @ Q
        is_valid, A0_valid = check_sign_restrictions_fast(A0_candidate)
        if is_valid:
            accepted_A0.append(A0_valid)
    end_time = time.time()
    acceptance_rate = len(accepted_A0) / attempts
    print('\nAlgorithm finished in ' + str(round(end_time - start_time, 2)) + ' seconds.')
    print('Total attempts: ' + str(attempts))
    print('Accepted draws: ' + str(len(accepted_A0)))
    print('Acceptance Rate: ' + str(round(acceptance_rate * 100, 4)) + '%')
    if len(accepted_A0) == 0:
        print('ERROR: No valid rotation matrices found. The data may not support the imposed sign restrictions.')
        sys.exit(1)
    horizon = 36
    all_Theta = np.zeros((len(accepted_A0), horizon + 1, K, K))
    for i, A0 in enumerate(accepted_A0):
        Phi = np.zeros((horizon + 1, K, K))
        Phi[0] = np.eye(K)
        for h in range(1, horizon + 1):
            for j in range(1, min(h, p) + 1):
                Phi[h] += B[j - 1] @ Phi[h - j]
        for h in range(horizon + 1):
            all_Theta[i, h] = Phi[h] @ A0
    all_Theta_accum = np.cumsum(all_Theta, axis=1)
    median_IRF_accum = np.percentile(all_Theta_accum, 50, axis=0)
    lower_68_accum = np.percentile(all_Theta_accum, 16, axis=0)
    upper_68_accum = np.percentile(all_Theta_accum, 84, axis=0)
    lower_95_accum = np.percentile(all_Theta_accum, 2.5, axis=0)
    upper_95_accum = np.percentile(all_Theta_accum, 97.5, axis=0)
    shock_names = ['Policy Shock', 'Demand Shock', 'Supply Shock']
    var_names = ['Mortgage Rate (pp)', 'Log Housing Starts', 'Log Home Prices']
    print('\n--- Accumulated IRFs (Median Responses) ---')
    horizons_to_print = [0, 6, 12, 24, 36]
    for j, shock in enumerate(shock_names):
        print('\n' + shock + ':')
        header = 'Horizon | ' + ' | '.join([v.ljust(18) for v in var_names])
        print(header)
        print('-' * len(header))
        for h in horizons_to_print:
            row = str(h).ljust(7) + ' | '
            row += ' | '.join([str(round(median_IRF_accum[h, i, j], 4)).ljust(18) for i in range(K)])
            print(row)
    plt.rcParams['text.usetex'] = False
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for i in range(K):
        for j in range(K):
            ax = axes[i, j]
            med = median_IRF_accum[:, i, j]
            l68 = lower_68_accum[:, i, j]
            u68 = upper_68_accum[:, i, j]
            l95 = lower_95_accum[:, i, j]
            u95 = upper_95_accum[:, i, j]
            ax.plot(range(horizon + 1), med, color='blue', linewidth=2)
            ax.fill_between(range(horizon + 1), l68, u68, color='blue', alpha=0.3, label='68% CI')
            ax.fill_between(range(horizon + 1), l95, u95, color='blue', alpha=0.1, label='95% CI')
            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            if i == 0:
                ax.set_title(shock_names[j])
            if j == 0:
                ax.set_ylabel(var_names[i])
            if i == 2:
                ax.set_xlabel('Months')
            ax.grid(True, alpha=0.3)
            if i == 0 and j == 2:
                ax.legend(loc='upper right')
    plt.tight_layout()
    timestamp = str(int(time.time()))
    plot_filename = 'data/structural_irfs_4_' + timestamp + '.png'
    plt.savefig(plot_filename, dpi=300)
    print('\nIRF summary plot saved to ' + plot_filename)
    out_path = 'data/structural_identification_results.npz'
    np.savez(out_path, accepted_A0=np.array(accepted_A0), all_Theta=all_Theta, all_Theta_accum=all_Theta_accum, B=np.array(B))
    print('Structural identification results saved to ' + out_path)