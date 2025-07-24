using ull = unsigned long long;
std::vector<ull> p(64);

void BasisInsert(ull x) {
    for (int i = 63; ~i; i--) {
        if (!(x >> i)) {
            continue;
        }
        if (!p[i]) {
            p[i] = x;
            break;
        }
        x ^= p[i];
    }
}

void solve() {
    int n;
    std::cin >> n;
    for (int i = 0; i < n; i++) {
        ull x;
        std::cin >> x;
        BasisInsert(x);
    }
    ull ans = 0;
    for (int i = 63; ~i; i--) {
        ans = std::max(ans, ans ^ p[i]);
    }
    std::cout << ans << "\n";
}