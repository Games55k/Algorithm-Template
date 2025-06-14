<img src="image.png" width="800" height="450">

<div style="text-align: center; font-weight: bold; font-size: 74px;">
  Algorithm Template
</div>

# 目录

- [数据结构](#数据结构)
- [数论](#数论)
- [图论](#图论)
- [字符串](#字符串)
- [STL](#STL)
- [杂项](#杂项)

# 数据结构

## 并查集

```cpp
struct DSU {
    std::vector<int> f, siz;

    DSU() {}
    DSU(int n) {
        init(n);
    }

    void init(int n) {
        f.resize(n);
        std::iota(f.begin(), f.end(), 0);
        siz.assign(n, 1);
    }
    int find(int x) {
        while (x != f[x]) {
            x = f[x] = f[f[x]];
        }
        return x;
    }
    bool same(int x, int y) {
        return find(x) == find(y);
    }
    bool merge(int x, int y) {
        x = find(x);
        y = find(y);
        if (x == y) {
            return false;
        }
        siz[x] += siz[y];
        f[y] = x;
        return true;
    }
    int size(int x) {
        return siz[find(x)];
    }
};
```

## 树状数组

```cpp

template <typename T>
struct Fenwick {
    int n;
    std::vector<T> a;

    Fenwick(int n_ = 0) {
        init(n_);
    }

    void init(int n_) {
        n = n_;
        a.assign(n, T{});
    }

    void add(int x, const T &v) {
        for (int i = x + 1; i <= n; i += i & -i) {
            a[i - 1] = a[i - 1] + v;
        }
    }

    T sum(int x) {
        T ans{};
        for (int i = x; i > 0; i -= i & -i) {
            ans = ans + a[i - 1];
        }
        return ans;
    }

    T rangeSum(int l, int r) {
        return sum(r) - sum(l);
    }
    
    int select(const T &k) {
        int x = 0;
        T cur{};
        for (int i = 1 << std::__lg(n); i; i /= 2) {
            if (x + i <= n && cur + a[x + i - 1] <= k) {
                x += i;
                cur = cur + a[x - 1];
            }
        }
        return x;
    }
};

```

## 线段树

### 区间加和单点修改

```cpp
//单点修改不需要lazy tag
struct Info {
    i64 cnt_e = 0, cnt_f = 0;
    i64 sum_e = 0, sum_f = 0;
    //维护区间e和f的个数和下标和
    Info() {}
    Info(i64 pos, char c) {
        //如果是e和f就单点修改信息
        if (c == 'e') {
            cnt_e = 1;
            sum_e = pos + 1;
        } else if (c == 'f') {
            cnt_f = 1;
            sum_f = pos + 1;
        }
    }
};

Info operator+(Info a, Info b) {
    //区间加法
    Info c;
    c.sum_e = a.sum_e + b.sum_e;
    c.cnt_e = a.cnt_e + b.cnt_e;

    c.sum_f = a.sum_f + b.sum_f;
    c.cnt_f = a.cnt_f + b.cnt_f;
    return c;
}

//维护base index 0的[l, r)区间
struct SegmentTree {
    int n;
    std::vector<int> tag;
    std::vector<Info> info;
    SegmentTree(int n_) : n(n_), tag(4 * n), info(4 * n) {}
    void pull(int p) {
        info[p] = info[2 * p] + info[2 * p + 1];
    }
    void add(int p, int v) {
        tag[p] += v;
    }
    void push(int p) {
        add(2 * p, tag[p]);
        add(2 * p + 1, tag[p]);
        tag[p] = 0;
    }
    
    Info query(int p, int l, int r, int x, int y) {
        if (l >= y || r <= x) {
            return {};
        }
        if (l >= x && r <= y) {
            return info[p];
        }
        int m = (l + r) / 2;
        push(p);
        return query(2 * p, l, m, x, y) + query(2 * p + 1, m, r, x, y);
    }
    
    Info query(int x, int y) {
        return query(1, 0, n, x, y);
    }
    
    void rangeAdd(int p, int l, int r, int x, int y, int v) {
        if (l >= y || r <= x) {
            return;
        }
        if (l >= x && r <= y) {
            return add(p, v);
        }
        int m = (l + r) / 2;
        push(p);
        rangeAdd(2 * p, l, m, x, y, v);
        rangeAdd(2 * p + 1, m, r, x, y, v);
        pull(p);
    }
    
    void rangeAdd(int x, int y, int v) {
        rangeAdd(1, 0, n, x, y, v);
    }
    
    void modify(int p, int l, int r, int x, const Info &v) {
        if (r - l == 1) {
            info[p] = v;
            return;
        }
        int m = (l + r) / 2;
        push(p);
        if (x < m) {
            modify(2 * p, l, m, x, v);
        } else {
            modify(2 * p + 1, m, r, x, v);
        }
        pull(p);
    }
    
    void modify(int x, const Info &v) {
        modify(1, 0, n, x, v);
    }
 };
```

## 珂朵莉树

```cpp
struct Chtholly {
    const int n;
    std::map<int, int> mp;
    Chtholly(int n) : n(n) { mp[-1] = 0; }

    void split(int x) {
        auto it = prev(mp.upper_bound(x));
        mp[x] = it->second;
    }
    void assign(int l, int r, int v) { // 注意，这里的r是区间右端点+1
        split(l);
        split(r);
        for (auto it = mp.find(l); it->first != r; it = mp.erase(it));
        mp[l] = v;
    }
    void update(int l, int r, int c) {
        split(l);
        split(r);
        for (auto it = mp.find(l); it->first != r; it = std::next(it)) {
            // 根据题目需要做些什么
        }
    }
};
```

# 数论

## 快速幂

```cpp
i64 binpow(i64 a, i64 b, i64 p) {
    i64 res = 1;
    while (b) {
        if (b & 1) res = res * a % p;
        a = a * a % p;
        b >>= 1;
    }
    return res;
}
```

## 欧拉筛
```cpp

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

```

## Exgcd

```cpp
int exgcd(int a, int b, i64 &x, i64 &y) {
    if (b == 0) {
        x = 1, y = 0;
        return a;
    }
    i64 g = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return g;
}
```

# 图论

## 最短路

### Floyd $(多源最短路)$

核心思想：枚举中转点

```cpp
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
```

#### Dijkstra (单源最短路)

堆优化版 ( $O(m\log m)$ ) 带路径
```cpp
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
```
## 最小生成树


### Prim算法( $O((n+m)\log n)$ )
思路：从一个点开始，每次不断加最小的点，从而确保每一个点到其它点都是最优

```cpp
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
```

## 倍增LCA

$dfs预处理时间复杂度(n \log n)，单次查询(\log n)$

```cpp

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

```

# 字符串

## 最长公共子序列 ( LCS ) ( $O(nm)$ )

思路：

```cpp
for (int i = 1; i <= s1.size(); i++) {
    for (int j = 1; j <= s2.size(); j++) {
        if (s1[i] == s2[j]) {
            dp[i][j] = dp[i - 1][j - 1] + 1;
        } else {
            dp[i][j] = std::max(dp[i - 1][j], dp[i][j - 1]);
        }
    }
}
```

## Trie树
```cpp
constexpr int N = 1e6 + 10;
int trie[N][26], cnt[N];
int tot = 0

void insert(std::string s) {
    int cur = 0;
    for (int i = 0; i < s.size(); i++) {
        int v = s[i] - 'a';
        if (!trie[cur][v]) {
            trie[cur][v] = ++tot;
        }
        cur = trie[cur][v];
    }
    cnt[cur]++;
}

int query(std::string s) {
    int cur = 0;
    for (int i = 0; i < s.size(); i++) {
        int v = s[i] - 'a';
        if (!trie[cur][v]) {
            return 0;
        }
        cur = trie[cur][v];
    }
    return cnt[cur];
}

```

## 01Trie

### 建树
```cpp
constexpr int N = 1 << 25;
int trie[N][2], cnt[N];
int tot = 0;

void insert(int x) {
    int cur = 0;
    for (int i = 30; i >= 0; i--) {
        int v = (x >> i) & 1;
        if (trie[cur][v] == -1) {
            trie[cur][v] = ++tot;
        }
        cur = trie[cur][v];
        cnt[cur]++;
    }
}
```

## 马拉车

思路：

```cpp
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
```

# STL

## 容器类

### 查找最大值

```cpp
auto max = *max_element(v.begin(), v.end());
auto max = std::ranges::max(v);  //C++20
```

```cpp
auto min = *min_element(v.begin(), v.end());
auto min = std::ranges::min(v);  //C++20
```

### 二分查找

```cpp
lower_bound(v.begin(), v.end(), target);  查找第一个大于等于target目标值的位置
upper_bound(v.begin(), v.end(), target); 查找第一个大于target目标值的位置
```

### 找第k小的数

```cpp
nth_element(v.begin(), v.begin() + k, v.end());
```

### 求和

```cpp
long long sum = accumulate(v.begin(), v.end(), 0ll);
```

### 翻转字符串

```cpp
reverse(s.begin(), s.end());
std::ranges::reverse(s);  //C++20
```

### 转换大小写
```cpp
std::transform(s.begin(), s.end(), s.begin(), ::toupper);
std::transform(s.begin(), s.end(), s.begin(), ::tolower);
```

### 排序

```cpp
sort(v.begin(), v.end());  默认升序
sort(v.begin(), v.end(), greater<>());  降序
std::ranges::sort(v);  //C++20
```

### 排列

```cpp
std::iota(v.begin(), v.end(), 0) //从0 ~ n - 1的排列

std::vector<int> v(4);
std::iota(v.begin(), v.end(), 0);
do {
    for (int i = 1; i <= 3; i++) {
        std::cout << v[i] << " \n"[i == 3];
    }
} while (std::next_permutation(v.begin() + 1, v.end())); //构造一个全排列

```

### 返回 $n$ 二进制下左边第一个 $1$ 的位置

```cpp
std::cout << std::__lg(n) << "\n";
```

### 平方和的平方根

```cpp
std::hypot(x, y); //勾股定理
std::hypot(x, y, z); //点到原点的距离 cpp17

```

## DEBUG类

```cpp
assert(条件);  //条件不成立时，程序终止
```

## 内建函数

```cpp
buitlin_popcount(x); //返回二进制下1的个数
```



# 杂项

## 顺时针九十度翻转矩阵

先把矩阵转置，然后每行翻转

```cpp
for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
        std::swap(matrix[i][j], matrix[j][i]);
    }
}
std::reverse(matrix.begin(), matrix.end());
```

## 枚举子集

复杂度 $O(n \cdot 2^n)$

```cpp
for (int i = 1; i < (1 << n); i++) {
    for (int j = 0; j < n; j++) {
        
        if (i & (1 << j))
        
    }
}
```