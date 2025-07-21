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