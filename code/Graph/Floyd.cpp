void Floyd() {
    for (int k = 1; k <= n; k++) {
        for (int i = 1; i <= n; i++) [
            for (int j = 1; j <= n; j++) {
                dp[i][j] = std::min(dp[i][j], dp[i][k] + dp[k][j]);
            }
        ]
    }
}

void solve() {
    std::vector dp(n + 1, std::vector<int>(n + 1, inf));
    while (m--) {
        int u, v, d;
        std::cin >> u >> v >> d;
        dp[u][v] = std::min(dp[u][v], d); 单向边
        dp[v][u] = std::min(dp[v][u], d); 双向边
    }
}