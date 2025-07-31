struct Comb {
    std::vector<i64> fac, invf;

    Comb() {}
    Comb(int n) {
        init(n + 1);
    }
    
    void init(int n) {
        fac.resize(n), invf.resize(n);
        fac[0] = invf[0] = 1;
        for (int i = 1; i < n; i++) {
            fac[i] = fac[i - 1] * i % mod;
            invf[i] = binpow(fac[i], mod - 2, mod);
        }
    }

    i64 binpow(i64 a, i64 b, i64 mod) {
        i64 res = 1;
        while (b) {
            if (b & 1) res = res * a % mod;
            a = a * a % mod;
            b >>= 1;
        }
        return res;
    }


    i64 C(int n, int k, int mod) {
        if (k < 0 || k > n) return 0;
        return fac[n] * invf[k] % mod * invf[n - k] % mod;
    }
};
