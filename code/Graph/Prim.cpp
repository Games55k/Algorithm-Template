void solve() {
    int n, m;
    std::cin >> n >> m;
    struct Edge {
        int v;
        i64 w;
        bool operator < (const Edge &x) const {
            return w > x.w;
        };
    };
    std::vector<std::vector<Edge>> g(n + 1);
    while (m--) {
        int u, v;
        i64 w;
        std::cin >> u >> v >> w;
        g[u].push_back({v, w});
        g[v].push_back({u, w});
    }
    std::vector<bool> vis(n + 1);
    i64 ans = 0;
    std::priority_queue<Edge> pq;
    pq.push({1, 0});
    while (!pq.empty()) {
        auto [x, w] = pq.top(); pq.pop();
        if (vis[x]) continue;
        vis[x] = true;
        ans += w;
        for (auto &[v, s] : g[x]) {
            pq.push({v, s});
        }
    }
    for (int i = 1; i <= n; i++) {
        if (!vis[i]) {
            std::cout << -1 << "\n";
            return;
        }
    }
    std::cout << ans << "\n";
}