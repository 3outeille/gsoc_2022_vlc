#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <limits>
#include <immintrin.h>

#include "dot_product.hh"
#include "dot_product_sse.hh"

using Clock = std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::microseconds;

struct TestCase
{
    float *input1;
    float *input2;
    size_t iteration;

    float get_random()
    {
        return (rand() / float(RAND_MAX)) * 10;
    }

    const size_t SIZE = 1024 * 1024 * 10;

    TestCase(size_t iteration_arg) : iteration(iteration_arg)
    {
        input1 = new float[SIZE];
        input2 = new float[SIZE];

        for (size_t i = 0; i < SIZE; i++)
        {
            input1[i] = get_random();
            input2[i] = get_random();
        }
    }

    ~TestCase()
    {
        delete[] input1;
        delete[] input2;
    }

    template <typename FUNCTION>
    void run(const char *name, FUNCTION func)
    {
        float result = 0.0;

        Clock::rep best_time = std::numeric_limits<Clock::rep>::max();

        for (size_t i = 0; i < iteration; i++)
        {
            const auto t1 = Clock::now();
            const float r = func(input1, input2, SIZE);
            const auto t2 = Clock::now();

            result = r;

            const Clock::rep dt = duration_cast<microseconds>(t2 - t1).count();

            if (dt < best_time)
                best_time = dt;
        }

        printf("%-20s: %lu us (result = %0.5f)\n", name, best_time, result);
    }
};

int main()
{
    TestCase test = TestCase(15);

    test.run("dot_product (naive)", dot_product);
    test.run("dot_product (sse)", dot_product_sse);
}