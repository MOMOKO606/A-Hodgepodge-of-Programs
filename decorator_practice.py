import functools
import math
import time


def display_time(threshold=0.2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"{func.__name__} took {end-start:.2f} seconds")
            if end - start > threshold: print(f"{func.__name__} took longer than {threshold} seconds")
            return result
        return wrapper
    return decorator

@display_time()
def count_primes_opt(n):
    primes = [True] * (n + 1)
    primes[0] = primes[1] = False
    for i in range(2, math.ceil(n ** 0.5) + 1):
        if not primes[i]:
            continue
        for j in range(i * 2, n + 1, i):
            primes[j] = False
    return sum(primes)

@display_time()
def count_primes_naive(n):
    if n < 2: return 0
    count = 0
    for i in range(2, n + 1):
        for j in range(2, i):
            if i % j == 0:
                break
        else:
            count += 1
    return count

def main():
    print(f"count_primes_opt's result: {count_primes_opt(10000)}")
    print(f"count_primes_naive's result: {count_primes_naive(10000)}")


if __name__ == "__main__":
    main()
