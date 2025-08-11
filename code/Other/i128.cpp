std::ostream& operator<<(std::ostream& os, i128 n) {
    if (n < 0) {
        os << "-"; n *= -1;
    }
    std::string s;
    while (n > 0) {
        s += char('0' + n % 10);
        n /= 10;
    }
    reverse(s.begin(), s.end());
    return os << s;
}
std::istream& operator>>(std::istream& is, i128& n) {
    std::string s; is >> s; n = 0;
    for (int i = 0; i < s.size(); i++) {
        n = n * 10 + s[i] - '0';
    }
    return is;
}