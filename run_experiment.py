from crowdsourcing import *
import pandas as pd
import random
from multiprocessing import Pool

NUM_CORES = 3
niter = 5
easiness_alpha=50
easiness_beta=1e-6
random.seed(285532)
id_vars = ["m", "n", "l", "r", "q", "easiness_alpha", "easiness_beta", "iternum"]  # used for pd.Melt()ing of results

## fig 1: vary problem size m and number of queries per task
results_1 = pd.DataFrame()
results_list_1 = []
with Pool(NUM_CORES) as pool:
    for iternum in range(niter):
        for l in [1, 3, 5, 6, 7, 9, 11, 13, 15]:  # these look like correct values from examination of fig. 1
            for m in [63, 250, 1000, 4000]:
                r = l
                n = int(m * l / r)
                q = 0.3
                # results_row = initialize_and_run_experiment(m=m, n=n, l=l, r=r, q=q, iternum=iternum, verbose=False)
                results_row = pool.apply_async(initialize_and_run_experiment, [m, n, l, r, q, iternum, easiness_alpha, easiness_beta, False])
                results_list_1.append(results_row)
    pool.close()
    pool.join()
for res in results_list_1:
    results_1 = results_1.append(res.get(), ignore_index=True)
pd.melt(results_1, id_vars=id_vars, var_name="method", value_name="error") \
    .to_csv("results_1.csv", index=False)

## FIG 2a: vary number of queries per task, l
results_2a = pd.DataFrame()
results_list_2a = []
with Pool(NUM_CORES) as pool:
    for iternum in range(niter):
        for l in range(2, 32, 3):
            m = 1000
            r = l
            n = int(m * l / r)
            q = 0.3
            # results_row = initialize_and_run_experiment(m=m, n=n, l=l, r=r, q=q, iternum=iternum, verbose=False)
            results_row = pool.apply_async(initialize_and_run_experiment, [m, n, l, r, q, iternum, easiness_alpha, easiness_beta, False])
            results_list_2a.append(results_row)
    pool.close()
    pool.join()
for res in results_list_2a:
    results_2a = results_2a.append(res.get(), ignore_index=True)
pd.melt(results_2a, id_vars=id_vars, var_name="method", value_name="error") \
    .to_csv("results_2a.csv", index=False)

## fig 2b: vary collective qualiry of crowd, q
results_2b = pd.DataFrame()
results_list_2b = []
with Pool(NUM_CORES) as pool:
    for iternum in range(niter):
        for q in np.linspace(0.001, 0.4, 12):
            l = 25  # l is fixed at 25 in this experiment
            m = 1000
            r = l
            n = int(m * l / r)
            # results_row = initialize_and_run_experiment(m=m, n=n, l=l, r=r, q=q, iternum=iternum, verbose=False)
            results_row = pool.apply_async(initialize_and_run_experiment, [m, n, l, r, q, iternum, easiness_alpha, easiness_beta, False])
            results_list_2b.append(results_row)
    pool.close()
    pool.join()
for res in results_list_2b:
    results_2b = results_2b.append(res.get(), ignore_index=True)
pd.melt(results_2b, id_vars=id_vars, var_name="method", value_name="error") \
    .to_csv("results_2b.csv", index=False)

# vary task difficulty
results_3 = pd.DataFrame()
results_list_3 = []
with Pool(NUM_CORES) as pool:
    for iternum in range(niter):
        for task_easiness_beta in [1e-6, 2.5, 5, 7.5, 10, 15, 20, 25]:
            m = 1000
            l = 25
            r = l
            q = 0.3
            n = int(m * l / r)
            results_row = pool.apply_async(initialize_and_run_experiment, [m, n, l, r, q, iternum, easiness_alpha, task_easiness_beta, False])
            results_list_3.append(results_row)
    pool.close()
    pool.join()
for res in results_list_3:
    results_3 = results_3.append(res.get(), ignore_index=True)
pd.melt(results_3, id_vars=id_vars, var_name="method", value_name="error") \
    .to_csv("results_3.csv", index=False)