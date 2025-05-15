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

### 合并
```cpp
void merge() { 
    p[find(x)] = find(y);
}
```

### 路径压缩
```cpp
int find(int n) { 
    return p[n] = (p[n] == n ? n : find(p[n]));
}
```


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

## 双端队列(deque)
```cpp
dq.front();  返回队首
dq.back();  返回队尾
dq.push_back(T);  队尾入队
dq.pop_back();  队尾出队
dq.push_front(T);  队首入队
dq.pop_front();  队首出队
dq.insert();  在指定位置前插入元素（传入迭代器和元素）
dq.erase();  删除指定位置的元素（传入迭代器）
dq.empty();  返回是否为空
dq.size();  返回元素数量
```
## 字符串
```cpp
str.find(ch, start = 0) 查找并返回从 start 开始的字符 ch 的位置
str.rfind(ch) 从末尾开始，查找并返回第一个找到的字符 ch 的位置（皆从 0 开始）（如果查找不到，返回 -1）
str.substr(start, len) 可以从字符串的 start（从 0 开始）截取一个长度为 len 的字符串（缺少 len 时代码截取到字符串末尾）。
str.append(s, pos, n) 将字符串 s 中，从 pos 开始的 n 个字符连接到当前字符串结尾。
str.replace(pos, n, s) 删除从 pos 开始的 n 个字符，然后在 pos 处插入串 s。
str.erase(pos, n) 删除从 pos 开始的 n 个字符。
str.insert(pos, s) 在 pos 位置插入字符串 s。
```


## 容器类

### 查找最大值

```cpp
long long max = *max_element(v.begin(), v.end());
long long max = std::ranges::max(v);  C++20
```

```cpp
long long min = *min_element(v.begin(), v.end());
long long min = std::ranges::min(v);  C++20
```

### 二分查找

```cpp
lower_bound(v.begin(), v.end(), target);  查找第一个大于等于target目标值的位置
upper_bound(v.begin(), v.end(), target); 查找第一个大于target目标值的位置
binary_search(v.begin(), v.end(), target); 查找target是否存在，找到返回true，否则返回false
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
std::ranges::reverse(s);  C++20
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
std::ranges::sort(v);  C++20
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
for (int i = 0; i < n; i++) {
    for (int j = 0; j < n / 2; j++) {
        std::swap(matrix[i][j], matrix[i][n - j - 1]);
    }
}
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