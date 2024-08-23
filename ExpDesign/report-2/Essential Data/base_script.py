import pygmo as pg
import time
import csv

import os

############ You should change the configuration below for the first objective #######

# number of problems to use from the benchmark set (max: 30):
n_probs = 8  # 4

# Number of repetitions per problem/algorithm pair:
n_reps = 194 # 384 # 96   # 3

##### You should change the configuration below for the second objective ##############

# DE Configuration (Change these for problem 2)
myF = 0.05   # Differential Weight, from 0 to 1!  DEFAULT 0.8
myCR = 0 # Crossover Probability, from 0 to 1  DEFAULT 0.9
# DONE 1, 0.9
##############################################################################
###### You should not change below unless you know what you are doing ########

# General algorithm params:
mgen = 1000 # 1000
popsize = 20 # 20

### Run Setup: ###

# Algorithm configuration: 1- Best, 2- Rand, 3- Rand-to-Best
algo1 = pg.de(gen=mgen, F=myF, CR=myCR, variant=1)
algo2 = pg.de(gen=mgen, F=myF, CR=myCR, variant=2)
algo3 = pg.de(gen=mgen, F=myF, CR=myCR, variant=3)
algos = [algo1, algo2, algo3]

# Problem set configuration
probs = []
for _p in range(n_probs):
    probs.append(pg.problem(pg.cec2014(prob_id=_p+1, dim=20)))


# Running the algorithm
timenow = time.strftime("%Y%m%d%H%M%S", time.localtime())

out_dir = f"./results/{timenow}-{n_probs}-{n_reps}-{myF}-{myCR}"

parent_out = os.path.join("/Users/mlnick/Documents/University/Git/Tsukuba-Master/ExpDesign/report-2/", out_dir)
os.mkdir(parent_out)

parent_out_all = os.path.join(parent_out, "all")
os.mkdir(parent_out_all)

parent_out_best = os.path.join(parent_out, "best")
os.mkdir(parent_out_best)

csvfile_all = open(f"{parent_out_all}/resultall_{timenow}.csv", "w", newline = '')
csvfile_best = open(f"{parent_out_best}/resultbest_{timenow}.csv", "w", newline = '')
csvwriter_a = csv.writer(csvfile_all)
csvwriter_a.writerow(["Selection Policy",f"Problem",f"Repetition","neval","best"])

csvwriter_b = csv.writer(csvfile_best)
csvwriter_b.writerow(["Selection Policy",f"Problem",f"Repetition","best"])

runtime_start = time.time()  # Record algorithm execution time

algo_runtime = []

for p_id, p in enumerate(probs):
    for a_id, a in enumerate(algos):
        algo_runtime.append(time.time())
        for r_id in range(n_reps):
            uda = a
            algo = pg.algorithm(uda)
            algo.set_verbosity(100)
            algo.set_seed(r_id)

            pop = pg.population(p, size = popsize)
            pop = algo.evolve(pop)

            # Extract the data
            log = algo.extract(type(uda)).get_log()
            for l in log:
                row = [a_id, p_id, r_id, l[1], l[2]]
                csvwriter_a.writerow(row)
            row = [a_id, p_id, r_id, l[2]]
            csvwriter_b.writerow(row)

runtime_stop = time.time()  # Record execution end timestamp
runtime = runtime_stop - runtime_start

algo1_runtime = algo_runtime[1] - algo_runtime[0]
algo2_runtime = algo_runtime[2] - algo_runtime[1]
algo3_runtime = algo_runtime[3] - algo_runtime[2]

runtime_str = f"Runtime with probs={n_probs}, reps={n_reps}, F={myF}, CR={myCR}: {runtime:.2f} seconds, alg1={algo1_runtime:.2f}s, alg2={algo2_runtime:.2f}s, alg3={algo3_runtime:.2f}s"

print(runtime_str)

csvwriter_a.writerow([runtime_str])
csvwriter_b.writerow([runtime_str])

csvfile_all.close()
csvfile_best.close()

# For convenient use in all_stats.py
print(f"dir = \"{timenow}-{n_probs}-{n_reps}-{myF}-{myCR}\"")
print(f"file = \"resultall_{timenow}.csv\"")

# dir = "20240715180434-8-24-0.7-0.75"
# file = "resultall_20240715180434.csv"

