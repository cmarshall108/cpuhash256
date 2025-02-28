import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

def calculate_avalanche_effect(hash_func, num_samples=1000, input_size=32):
    """
    Test the avalanche effect by flipping single bits and measuring hash changes.
    Returns: average percentage of output bits that change
    """
    changes = []
    
    for _ in range(num_samples):
        # Generate random input
        input_data = np.random.bytes(input_size)
        original_hash = hash_func(input_data)
        
        # Test each bit flip
        for byte_idx in range(len(input_data)):
            for bit_idx in range(8):
                # Flip a single bit
                modified = bytearray(input_data)
                modified[byte_idx] ^= (1 << bit_idx)
                
                # Calculate new hash
                new_hash = hash_func(bytes(modified))
                
                # Count bit differences
                diff_count = sum(bin(a ^ b).count('1') 
                               for a, b in zip(original_hash, new_hash))
                changes.append(diff_count / (len(original_hash) * 8) * 100)
    
    return np.mean(changes), np.std(changes)

def test_bit_distribution(hash_func, num_samples=10000, input_size=32):
    """
    Test the distribution of bits in hash outputs.
    Returns: chi-square test statistic and p-value
    """
    bit_counts = defaultdict(int)
    total_bits = 0
    
    for _ in range(num_samples):
        input_data = np.random.bytes(input_size)
        hash_value = hash_func(input_data)
        
        for byte in hash_value:
            for i in range(8):
                bit = (byte >> i) & 1
                bit_counts[bit] += 1
                total_bits += 1
    
    # Chi-square test
    expected = total_bits / 2  # Expected count for each bit value
    chi_square = sum((count - expected) ** 2 / expected 
                    for count in bit_counts.values())
    
    return chi_square

def test_collision_resistance(hash_func, num_samples=100000, input_size=32):
    """
    Test for hash collisions using birthday attack probability estimation.
    Returns: number of collisions found
    """
    seen_hashes = set()
    collisions = 0
    
    for _ in range(num_samples):
        input_data = np.random.bytes(input_size)
        hash_value = hash_func(input_data)
        
        if hash_value in seen_hashes:
            collisions += 1
        seen_hashes.add(hash_value)
    
    return collisions

def plot_security_results(results, title, output_file):
    """Plot security test results"""
    algorithms = list(results.keys())
    metrics = ['Avalanche Effect (%)', 'Bit Distribution χ²', 'Collisions']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)
    
    for i, metric in enumerate(metrics):
        values = [results[algo][i] for algo in algorithms]
        axes[i].bar(algorithms, values)
        axes[i].set_title(metric)
        axes[i].set_xticklabels(algorithms, rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
