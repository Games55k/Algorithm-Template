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