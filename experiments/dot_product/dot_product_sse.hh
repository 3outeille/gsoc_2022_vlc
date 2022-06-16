float dot_product_sse(const float *a, const float *b, size_t n)
{
    __m128 dots = _mm_setzero_ps();
    for (size_t i = 0; i < n; i += 4)
    {
        __m128 A = _mm_loadu_ps(&a[i]);
        __m128 B = _mm_loadu_ps(&b[i]);
        __m128 AB = _mm_mul_ps(A, B);

        dots = _mm_add_ps(dots, AB);
    }

    float tmp[4];
    _mm_storeu_ps(tmp, dots);
    return dots[0] + dots[1] + dots[2] + dots[3];
}