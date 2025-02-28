#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Platform-specific headers and defines
#if defined(__x86_64__) && defined(__AVX512F__) && defined(__AVX512DQ__) && !defined(__aarch64__) && !defined(__ARM_NEON)
#include <immintrin.h>
#define USE_AVX512
#elif defined(__aarch64__) || (defined(__APPLE__) && defined(__ARM_NEON)) // macOS ARM support
#include <arm_neon.h>
#define USE_NEON
#else
#define USE_SCALAR
#endif

#define BLOCK_SIZE 64    // 64 bytes (512 bits)
#define STATE_SIZE 1024  // 1024-bit state

// Rotate macro for scalar
#define ROTR64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))

// State definition
#ifdef USE_AVX512
typedef struct {
    __m512i part1;  // 512 bits
    __m512i part2;  // 512 bits
} state_t;
#elif defined(USE_NEON)
typedef struct {
    uint64x2_t part[8];  // 8x128-bit = 1024 bits
} state_t;
#else  // USE_SCALAR
typedef struct {
    uint64_t data[16];  // 1024 bits
} state_t;
#endif

// Initial state constants
static const uint64_t INIT_STATE[16] = {
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
    0xdeadbeefdeadbeef, 0xfeedfacefeedface, 0xcafebabebaadf00d, 0xabad1deaabad1dea,
    0x123456789abcdef0, 0xabcdef0123456789, 0x55555555aaaaaaaa, 0xaaaaaaaa55555555
};

// Mixing functions
#ifdef USE_AVX512
static inline void mix(state_t* S, __m512i m1, __m512i m2) {
    S->part1 = _mm512_add_epi64(S->part1, m1);
    S->part2 = _mm512_xor_si512(S->part2, S->part1);
    S->part1 = _mm512_mullo_epi64(S->part1, _mm512_set1_epi64(0x9e3779b97f4a7c15));
    __m512i perm_idx = _mm512_and_si512(m1, _mm512_set1_epi64(7));
    S->part2 = _mm512_permutexvar_epi64(perm_idx, S->part2);
    S->part2 = _mm512_rol_epi64(S->part2, 17);
    S->part1 = _mm512_mullo_epi64(S->part1, S->part2);
}
#elif defined(USE_NEON)
static inline void mix(state_t* S, uint64x2_t m1, uint64x2_t m2, uint64_t r) {
    uint64x2_t konst = vdupq_n_u64(0x9e3779b97f4a7c15);
    int64x2_t shift_vec = vdupq_n_s64(r & 63);           // Positive shift for left
    int64x2_t neg_shift_vec = vdupq_n_s64(-(r & 63));    // Negative shift for right
    for (int i = 0; i < 8; i += 2) {
        S->part[i] = vaddq_u64(S->part[i], (i < 4) ? m1 : m2);
        S->part[i] = veorq_u64(S->part[i], S->part[i + 1]);
        S->part[i + 1] = vaddq_u64(S->part[i + 1], konst);
        S->part[i] = vorrq_u64(vshlq_u64(S->part[i], shift_vec), vshlq_u64(S->part[i], neg_shift_vec));
        S->part[i + 1] = veorq_u64(S->part[i + 1], S->part[i]);
    }
}
#else  // USE_SCALAR
static inline void mix(state_t* S, uint64_t m1, uint64_t m2, uint64_t r) {
    const uint64_t konst = 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < 8; i++) {
        S->data[i] += (i < 4) ? m1 : m2;
        S->data[i + 8] ^= S->data[i];
        S->data[i] *= konst;
        S->data[i + 8] = ROTR64(S->data[i + 8], r & 63);
        S->data[i] *= S->data[i + 8];
    }
}
#endif

// Compression function (processes 128-byte blocks)
static void compress(state_t* H, const uint8_t* block) {
#ifdef USE_AVX512
    __m512i m1 = _mm512_loadu_si512((const __m512i*)block);
    __m512i m2 = _mm512_loadu_si512((const __m512i*)(block + 64));
    mix(H, m1, m2);
#elif defined(USE_NEON)
    uint64x2_t m1 = vld1q_u64((const uint64_t*)block);
    uint64x2_t m2 = vld1q_u64((const uint64_t*)(block + 64));
    uint64_t r = ((const uint64_t*)block)[0];
    mix(H, m1, m2, r);
#else  // USE_SCALAR
    uint64_t m1 = ((const uint64_t*)block)[0];
    uint64_t m2 = ((const uint64_t*)(block + 64))[0];
    uint64_t r = m1;
    mix(H, m1, m2, r);
#endif
}

// Finalization step
static void finalize(state_t* H, uint64_t len) {
#ifdef USE_AVX512
    __m512i len_vec = _mm512_set1_epi64(len);
    __m512i const_vec = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF);
    mix(H, len_vec, const_vec);
#elif defined(USE_NEON)
    uint64x2_t len_vec = vdupq_n_u64(len);
    uint64x2_t const_vec = vdupq_n_u64(0xFFFFFFFFFFFFFFFF);
    mix(H, len_vec, const_vec, len);
#else  // USE_SCALAR
    mix(H, len, 0xFFFFFFFFFFFFFFFF, len);
#endif
}

// Padding function
static void pad_message(const uint8_t* input, size_t len, uint8_t** padded, size_t* padded_len) {
    size_t len_bits = len * 8;
    size_t total_blocks = (len + 1 + 8 + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);  // 128-byte chunks
    *padded_len = total_blocks * BLOCK_SIZE * 2;

    *padded = (uint8_t*)malloc(*padded_len);
    if (!*padded) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    memcpy(*padded, input, len);
    (*padded)[len] = 0x80;  // Append '1' bit
    memset(*padded + len + 1, 0, *padded_len - len - 9);
    for (int i = 0; i < 8; i++) {
        (*padded)[*padded_len - 8 + i] = (len_bits >> (56 - i * 8)) & 0xFF;  // Append length
    }
}

// Main hash function
void cpuhash256(const uint8_t* message, size_t len, uint8_t* hash) {
    state_t H;
#ifdef USE_AVX512
    H.part1 = _mm512_loadu_si512((const __m512i*)INIT_STATE);
    H.part2 = _mm512_loadu_si512((const __m512i*)(INIT_STATE + 8));
#elif defined(USE_NEON)
    for (int i = 0; i < 8; i++) {
        H.part[i] = vld1q_u64(&INIT_STATE[i * 2]);
    }
#else  // USE_SCALAR
    memcpy(H.data, INIT_STATE, sizeof(H.data));
#endif

    uint8_t* padded = NULL;  // Dynamic buffer
    size_t padded_len;
    pad_message(message, len, &padded, &padded_len);

    for (size_t i = 0; i < padded_len; i += BLOCK_SIZE * 2) {
        compress(&H, padded + i);
    }

    finalize(&H, len);

#ifdef USE_AVX512
    _mm256_storeu_si256((__m256i*)hash, _mm512_extracti64x4_epi64(H.part1, 0));
#elif defined(USE_NEON)
    for (int i = 0; i < 4; i++) {
        vst1q_lane_u64((uint64_t*)(hash + i * 8), H.part[i], 0);
    }
#else  // USE_SCALAR
    memcpy(hash, H.data, 32);
#endif

    free(padded);  // Free dynamically allocated memory
}