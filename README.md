# CPUHash256

This is the test suite (also includes the CPUHash256 C implementation) for Python benchmark and testing

## Building
```
python3 setup.py build_ext --inplace
```

### Installing
```
pip uninstall cpuhash256  # Remove any existing installations
pip install -e .  # Install in development mode
```

### Running Tests
```
python3 hash_performance_test.py
```

## Benchmark tests

![Benchmark #1](hash_performance_hash_performance_-_alternate.png "Benchmark #1")
![Benchmark #2](hash_performance_hash_performance_-_increasing.png "Benchmark #2")
![Benchmark #3](hash_performance_hash_performance_-_ones.png "Benchmark #3")
![Benchmark #4](hash_performance_hash_performance_-_random.png "Benchmark #4")
![Benchmark #5](hash_performance_hash_performance_-_zeros.png "Benchmark #5")

## Security tests

![Security #1](hash_security_comparison.png "Security Test #1")