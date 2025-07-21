void solve() {
    int n;
    std::cin >> n;
    std::vector<int> dep(n + 1);
    std::vector<std::vector<int>> g(n + 1);
    std::vector<std::array<int, 21>> p(n + 1); //p[i][j]表示节点i走2^j步能到的节点
    for (int i = 2, x; i <= n; i++) {
        std::cin >> x;
        p[i][0] = x;
        g[x].push_back(i);
    }

    auto dfs = [&](auto &&self, int x) -> void {
        dep[x] = dep[p[x][0]] + 1;
        for (int i = 1; i <= 20; i++) {
            p[x][i] = p[p[x][i - 1]][i - 1];
            //预处理出所有p[i][j]
        }
        for (auto &v : g[x]) {
            self(self, v);
        }
    };

    dfs(dfs, 1);

    auto lca = [&](int u, int v) -> int {
        if (dep[u] < dep[v]) {
            std::swap(u, v);
        }
        for (int i = 20; i >= 0; i--) {
            if (dep[p[u][i]] >= dep[v]) {
                u = p[u][i];
            }
        }
        if (u == v) return u;
        for (int i = 20; i >= 0; i--) {
            if (p[u][i] != p[v][i]) {
                u = p[u][i], v = p[v][i];
            }
        }
        return p[u][0];
    };

    int q;
    std::cin >> q;
    while (q--) {
        int u, v;
        std::cin >> u >> v;
        std::cout << lca(u, v) << "\n";
    }
}