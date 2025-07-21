//1.区间加和单点修改

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