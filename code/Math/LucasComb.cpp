auto binpow = [](i64 a, i64 b, i64 p) -> i64 {
        i64 res = 1ll;
        while (b) {
            if (b & 1) res = res * a % p;
            a = a * a % p;
            b >>= 1;
        }
        return res;
};

i64 C(i64 a, i64 b, i64 p) {
    if (b > a) {
        return 0;
    }
    i64 res = 1;
    for(i64 i = 1, j = a; i <= b; i++, j--)
    {
        res = res * j % p;
        res=res * binpow(i, p - 2, p) % p;
    }
    return res;
}
i64 lucas(i64 a, i64 b, i64 p) {
    if(a < p && b < p) {
        return C(a, b, p);
    }
    return (C(a % p, b % p, p) * lucas(a / p, b / p, p)) % p;
}