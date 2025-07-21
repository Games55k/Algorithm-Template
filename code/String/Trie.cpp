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