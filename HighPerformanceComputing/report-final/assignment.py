def compute_parallel_metrics(N_list):
    """
    Given a list of processor counts, compute:
    - T(N): total execution time
    - Speedup(N)
    - E(N): parallel efficiency (%)
    - P(N): effective performance in GFLOPS
    """
    
    # Single-processor reference time
    T_single = 500.0  # seconds
    
    # Total FLOPs for the problem (1 TFLOP = 1000 GFLOP)
    total_flops_gf = 1000.0
    
    results = []
    for N in N_list:
        # Parallel computation time
        T_comp = T_single / N
        # Communication time
        T_comm = 0.2 * (N ** 2)
        # Total time
        T_total = T_comp + T_comm
        
        # Speedup relative to single processor
        speedup = T_single / T_total
        
        # Parallel efficiency in %
        efficiency = (speedup / N) * 100
        
        # Effective performance in GFLOPS
        perf_gflops = total_flops_gf / T_total
        
        results.append((N, T_total, speedup, efficiency, perf_gflops))
    return results

# Example usage:
N_values = [2, 5, 10, 100]
metrics = compute_parallel_metrics(N_values)

# Print the results in a formatted table
print(f"{'N':>5} | {'T(N) (s)':>10} | {'Speedup':>8} | {'E(N) (%)':>9} | {'P(N) (GFLOPS)':>14}")
print("-"*60)
for (N, T_val, sp, eff, perf) in metrics:
    print(f"{N:5d} | {T_val:10.1f} | {sp:8.3f} | {eff:9.3f} | {perf:14.3f}")


import numpy as np
import time

def test_fft_performance(N=65536, num_runs=5):
    """
    Times the execution of an N-point complex-to-complex FFT using NumPy.
    Returns the average time over num_runs runs.
    """

    # Create a random complex input 
    x = np.random.rand(N) + 1j * np.random.rand(N)

    # Pre-warm the FFT to avoid measuring any first-call overhead
    _ = np.fft.fft(x)

    # Measure execution times
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        y = np.fft.fft(x)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / num_runs
    return avg_time

if __name__ == "__main__":
    N = 65536
    avg_time = test_fft_performance(N)
    print(f"Average execution time for {N}-point FFT: {avg_time:.6f} seconds")

    import math

cache_size_bytes = int(1.5 * 1024 * 1024)  # 1.5 MiB
element_size = 8  # 8 bytes per double

# We want 3 blocks: A_block, B_block, C_block
# So total bytes = 3 * B^2 * element_size
# Solve 3 * B^2 * element_size <= cache_size_bytes
B_max = int(math.sqrt(cache_size_bytes / (3.0 * element_size)))
print("Computed maximum B =", B_max)

N = 131072
ways = 12
line_size = 64
element_size = 8

# Each row should be a multiple of ways * line_size / element_size = 96 elements
aligned_unit = (ways * line_size) // element_size  # 768 bytes / 8 bytes = 96
remainder = N % aligned_unit
p = (aligned_unit - remainder) if remainder != 0 else 0

print("N mod 96 =", remainder)
print("Needed padding p =", p)
print("New row length =", N + p)