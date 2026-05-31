#ifndef __RANDOM_INT_GENERATOR__
#define __RANDOM_INT_GENERATOR__


class RandomIntGenerator {
    public:
        RandomIntGenerator(int seed, int min, int max):
            _seed(seed),
            _last(seed),
            _min(min),
            _max(max) { };

        int operator()() {
            _last = (1091 * _last + 1093) % 199;
            return _last % (_max - _min) + _min; 
        }

    private:
        int _last;
        int _seed;
        int _min;
        int _max;
};

#endif //__RANDOM_INT_GENERATOR__