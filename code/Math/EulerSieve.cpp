std::vector<int> min_prime, primes;

void sieve(int n) {
    min_prime.assign(n + 1, 0);
    primes.clear();
    
    for (int i = 2; i <= n; i++) {
        if (min_prime[i] == 0) {
            min_prime[i] = i;
            primes.push_back(i);
        }
        
        for (auto p : primes) {
            if (i * p > n) {
                break;
            }
            min_prime[i * p] = p;
            if (p == min_prime[i]) {
                break;
            }
        }
    }
}