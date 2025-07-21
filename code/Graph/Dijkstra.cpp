void solve() {
    int n, m;
    std::cin >> n >> m;
    struct Edge{
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
    std::vector<i64> dis(n + 1, LLONG_MAX);
    std::vector<bool> vis(n + 1);
    std::vector<int> pre(n + 1, -1);
    auto dijkstra = [&](int st) {
        dis[st] = 0;
        std::priority_queue<Edge> pq;
        pq.push({st, dis[st]});
        while (!pq.empty()) {
            int x = pq.top().v; pq.pop();
            if (vis[x]) continue;
            vis[x] = true;
            for (auto &[v, w] : g[x]) {
                if (!vis[v] && dis[x] + w < dis[v]) {
                    pre[v] = x;
                    dis[v] = dis[x] + w;
                    pq.push({v , dis[v]});
                }
            }
        }
    };
    dijkstra(1);
    if (dis[n] != LLONG_MAX) {
        std::vector<int> ans;
        int idx = n;
        while (pre[idx] != 1) {
            ans.push_back(idx);
            idx = pre[idx];
        }
        ans.push_back(idx);
        ans.push_back(1);
        std::ranges::reverse(ans);
        for (auto &i : ans) {
            std::cout << i << " \n"[&i == &ans.back()];
        }
        std::cout << dis[n] << "\n";
    } else {
        std::cout << -1 << "\n";
    }
}