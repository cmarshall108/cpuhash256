import hashlib
import time
import numpy as np
import matplotlib.pyplot as plt
from cpuhash256 import hash256
from hashlib import blake2b
from blake3 import blake3
import itertools
import concurrent.futures

def generate_test_data(size_kb, pattern='random'):
    """Memory efficient test data generator"""
    size_bytes = size_kb * 1024
    if pattern == 'random':
        # Generate random data in chunks
        chunk_size = min(size_bytes, 1024 * 1024)  # 1MB chunks
        remaining = size_bytes
        while remaining > 0:
            current_chunk = min(remaining, chunk_size)
            yield np.random.bytes(current_chunk)
            remaining -= current_chunk
    else:
        # Use repeating patterns without storing full data
        if pattern == 'zeros':
            chunk = b'\x00' * min(size_bytes, 1024)
        elif pattern == 'ones':
            chunk = b'\xff' * min(size_bytes, 1024)
        elif pattern == 'alternate':
            chunk = b'\x55' * min(size_bytes, 1024)
        elif pattern == 'increasing':
            chunk = bytes(range(min(size_bytes, 256)))
        
        remaining = size_bytes
        while remaining > 0:
            current_chunk = min(remaining, len(chunk))
            yield chunk[:current_chunk]
            remaining -= current_chunk

def benchmark_hash(hash_func, data_generator, size_kb, iterations=100):
    """Memory efficient benchmark"""
    times = []
    try:
        # Warmup run with generator
        h = hash_func(b''.join(data_generator()))
        
        for _ in range(iterations):
            start = time.perf_counter()
            h = hash_func(b''.join(data_generator()))
            times.append(time.perf_counter() - start)
        return np.mean(times), np.std(times)
    except Exception as e:
        print(f"Error in {hash_func.__name__}: {str(e)}")
        return float('inf'), float('inf')

def test_sha256(data):
    return hashlib.sha256(data).digest()

def test_blake2b(data):
    return blake2b(data).digest()

def test_blake3(data):
    return blake3(data).digest()

def test_cpuhash256(data):
    """Wrapper to make cpuhash256 return a digest-compatible result"""
    result = hash256(data)
    # Ensure the result is in the correct format (bytes)
    if not isinstance(result, bytes) or len(result) != 32:
        raise ValueError("cpuhash256 returned invalid hash format")
    return result

def run_parallel_tests(hash_func, data_list, iterations=100):
    """Run hash tests in parallel"""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(benchmark_hash, hash_func, data, iterations) 
                  for data in data_list]
        return [f.result() for f in futures]

def plot_results(sizes, results, title):
    plt.figure(figsize=(10, 6))
    algorithms = ['SHA256', 'Blake2b', 'CPUHash256']
    
    for i, algo in enumerate(algorithms):
        means = [r[i][0] for r in results]
        stds = [r[i][1] for r in results]
        plt.errorbar(sizes, means, yerr=stds, label=algo, marker='o')
    
    plt.xlabel('Input Size (KB)')
    plt.ylabel('Time (seconds)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'hash_performance_{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_pattern_comparison(sizes, patterns, results, title):
    """Plot performance comparison for different data patterns"""
    plt.figure(figsize=(12, 8))
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, pattern in enumerate(patterns):
        means = [r[i][0] for r in results]
        stds = [r[i][1] for r in results]
        plt.errorbar(sizes, means, yerr=stds, 
                    label=f'{pattern}', marker=markers[i % len(markers)])
    
    plt.xlabel('Input Size (KB)')
    plt.ylabel('Time (seconds)')
    plt.title(f'{title} - Pattern Comparison')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(f'hash_pattern_{title.lower().replace(" ", "_")}.png')
    plt.close()

def get_available_hashlib_algorithms():
    """Get all available hashlib algorithms"""
    return sorted([algo for algo in hashlib.algorithms_available 
                  if not algo.startswith('shake')])  # exclude variable length hashes

def create_hash_function(algo_name):
    """Create a hash function wrapper for any algorithm"""
    def hash_func(data):
        if algo_name == 'blake3':
            return blake3(data).digest()
        elif algo_name == 'cpuhash256':
            return hash256(data)
        else:
            return hashlib.new(algo_name, data).digest()
    hash_func.__name__ = algo_name
    return hash_func

def main():
    print("Starting comprehensive hash algorithm comparison...")
    
    # Get all available hash functions
    algorithms = get_available_hashlib_algorithms() + ['blake3', 'cpuhash256']
    print(f"Testing {len(algorithms)} hash algorithms: {', '.join(algorithms)}")
    
    # Test configurations
    sizes = [1, 4, 16, 64, 256]  # KB
    patterns = ['random', 'zeros', 'ones', 'alternate', 'increasing']
    iterations = 50  # Reduced iterations due to more algorithms
    
    # Process one pattern at a time
    for pattern in patterns:
        print(f"\nTesting pattern: {pattern}")
        results = []
        
        for size in sizes:
            print(f"\n  Size: {size}KB")
            data_gen = lambda: generate_test_data(size, pattern)
            
            # Test all hash functions
            pattern_results = []
            for algo in algorithms:
                hash_func = create_hash_function(algo)
                result = benchmark_hash(hash_func, data_gen, size, iterations)
                pattern_results.append(result)
                print(f"    {algo:12s}: {result[0]*1e6:9.2f} μs (±{result[1]*1e6:7.2f})")
            
            results.append(pattern_results)
        
        # Plot results for this pattern
        plot_pattern_results(sizes, results, algorithms, f"Hash Performance - {pattern}")
        results.clear()

def plot_pattern_results(sizes, results, algorithms, title):
    """Memory efficient plotting with multiple algorithms"""
    plt.figure(figsize=(15, 10))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, algo in enumerate(algorithms):
        means = [r[i][0] for r in results]
        stds = [r[i][1] for r in results]
        plt.errorbar(sizes, means, yerr=stds, 
                    label=algo, 
                    marker=markers[i % len(markers)],
                    linestyle='-' if algo in ['sha256', 'blake2b', 'blake3', 'cpuhash256'] else '--')
    
    plt.xlabel('Input Size (KB)')
    plt.ylabel('Time (seconds)')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(f'hash_performance_{title.lower().replace(" ", "_")}.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
