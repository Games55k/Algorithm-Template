// 双哈希
using u64 = unsigned long long;
constexpr u64 base1 = 131, base2 = 171;
constexpr u64 mod1 = 1e9 + 7, mod2 = 1e9 + 9;

std::vector<u64> bs1(n + 1, 1), bs2(n + 1, 1), hash1(n + 1), hash2(n + 1);
auto GetHash1 = [&](int l, int r) -> u64 {
    return (hash1[r] - (hash1[l - 1] * bs1[r - l + 1]) % mod1 + mod1) % mod1;
};
auto GetHash2 = [&](int l, int r) -> u64 {
    return (hash2[r] - (hash2[l - 1] * bs2[r - l + 1]) % mod2 + mod2) % mod2;
};
for (int i = 0; i < n; i++) {
    bs1[i + 1] = (bs1[i] * base1) % mod1;
    bs2[i + 1] = (bs2[i] * base2) % mod2;
    hash1[i + 1] = ((hash1[i] * base1) % mod1 + s[i]) % mod1;
    hash2[i + 1] = ((hash2[i] * base2) % mod2 + s[i]) % mod2;
}