template<typename T>
struct Fraction {
    T x, y;
    Fraction(T a = 0, T b = 1) {
        if(b < 0) a = -a, b = -b;
        x = a, y = b;
    }
    bool operator < (const Fraction& other) const {
        return (*this).x * other.y < (*this).y * other.x;
    }
    bool operator <= (const Fraction &other) const {
        return (*this).x * other.y <= (*this).y * other.x;
    }
    bool operator == (const Fraction &other) const {
        return (*this).x * other.y == (*this).y * other.x;
    }
    bool operator > (const Fraction &other) const {
        return (*this).x * other.y > (*this).y * other.x;
    }
    bool operator >= (const Fraction &other) const {
        return (*this).x * other.y >= (*this).y * other.x;
    }
    bool operator != (const Fraction& other) const {
        return !(*this == other); 
    }
    Fraction operator + (const Fraction &other) const {
        return Fraction((*this).x * other.y + (*this).y * other.x, (*this).y * other.y);
    }
    Fraction operator - (const Fraction &other) const {
        return Fraction((*this).x * other.y - (*this).y * other.x, (*this).y * other.y);
    }
    Fraction operator * (const Fraction &other) const {
        return Fraction((*this).x * other.x, (*this).y * other.y);
    }
    Fraction operator / (const Fraction &other) const {
        return Fraction((*this).x * other.y, (*this).y * other.x);
    }
    Fraction inv() const { 
        return Fraction(y, x); 
    }
    static constexpr Fraction inf() noexcept { return Fraction(1, 0); }
};
using Frac = Fraction<long long>;