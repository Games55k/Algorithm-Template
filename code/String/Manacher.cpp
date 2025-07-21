int n;
std::cin >> n;
std::vector<int> p(2 * n + 3);
char s[2 * n + 3];
std::cin >> s + 1;
for (int i = 2 * n + 1; i >= 1; i--) {
    i & 1 ? s[i] = '#' : s[i] = s[i >> 1];
}
s[0] = '$', s[2 * n + 2] = '@';
int C = 0, R = 0, ans = 0;
for (int i = 1; i <= 2 * n + 1; i++) {
    p[i] = i < R ? std::min(p[2 * C - i], R - i) : 1;
    while (s[i + p[i]] == s[i - p[i]]) p[i]++;
    if (i + p[i] > R) C = i, R = i + p[i];
    ans = std::max(ans, p[i] - 1);
}
std::cout << ans << "\n";