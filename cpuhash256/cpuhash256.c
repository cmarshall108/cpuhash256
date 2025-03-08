#include <stdint.h>
#include <string.h>
#include <stdio.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// Type definitions
typedef uint32_t u32;
typedef uint8_t u8;
typedef uint64_t u64;

// Rotate right macro
#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

// Initial Vector (IV) - 512 bits (16 x 32-bit words)
static const u32 IV[16] = {
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
    0xC1059ED8, 0x367CD507, 0x3070DD17, 0xF70E5939,
    0xFFC00B31, 0x68581511, 0x64F98FA7, 0xBEFA4FA4
};

// SIMD-Optimized Block Loading
#ifdef __AVX2__
static const __m128i lane_mask = _mm_setr_epi8(3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12);
static const __m256i swap_mask = _mm256_set_m128i(lane_mask, lane_mask);

static void load_block_avx2(const u8 *input, u32 block[16]) {
    __m256i v0 = _mm256_loadu_si256((__m256i*)input);
    __m256i v1 = _mm256_loadu_si256((__m256i*)(input + 32));
    v0 = _mm256_shuffle_epi8(v0, swap_mask);
    v1 = _mm256_shuffle_epi8(v1, swap_mask);
    _mm256_storeu_si256((__m256i*)block, v0);
    _mm256_storeu_si256((__m256i*)(block + 8), v1);
}
#endif

#ifdef __ARM_NEON
static void load_block_neon(const u8 *input, u32 block[16]) {
    uint32x4_t v0 = vld1q_u32((const uint32_t*)input);
    uint32x4_t v1 = vld1q_u32((const uint32_t*)(input + 16));
    uint32x4_t v2 = vld1q_u32((const uint32_t*)(input + 32));
    uint32x4_t v3 = vld1q_u32((const uint32_t*)(input + 48));
    v0 = vrev32q_u8(v0);
    v1 = vrev32q_u8(v1);
    v2 = vrev32q_u8(v2);
    v3 = vrev32q_u8(v3);
    vst1q_u32(block, v0);
    vst1q_u32(block + 4, v1);
    vst1q_u32(block + 8, v2);
    vst1q_u32(block + 12, v3);
}
#endif

// Permutation Function with Platform-Specific Implementations
#if defined(__AVX2__)
// Define quarter_round for AVX2 since permute uses it
static inline void quarter_round(u32 *a, u32 *b, u32 *c, u32 *d) {
    *a += *b; *d = ROTR(*d ^ *a, 17);
    *c += *d; *b = ROTR(*b ^ *c, 13);
    *a += *b; *d = ROTR(*d ^ *a, 9);
    *c += *d; *b = ROTR(*b ^ *c, 5);
}

static void permute(u32 state[16]) {
    for (int round = 0; round < 5; round++) {
        quarter_round(&state[0], &state[4], &state[8], &state[12]);
        quarter_round(&state[1], &state[5], &state[9], &state[13]);
        quarter_round(&state[2], &state[6], &state[10], &state[14]);
        quarter_round(&state[3], &state[7], &state[11], &state[15]);
        quarter_round(&state[0], &state[5], &state[10], &state[15]);
        quarter_round(&state[1], &state[6], &state[11], &state[12]);
        quarter_round(&state[2], &state[7], &state[8], &state[13]);
        quarter_round(&state[3], &state[4], &state[9], &state[14]);
    }
}
#elif defined(__ARM_NEON)
// Define quarter_round for NEON since permute uses it
static inline void quarter_round(u32 *a, u32 *b, u32 *c, u32 *d) {
    *a += *b; *d = ROTR(*d ^ *a, 17);
    *c += *d; *b = ROTR(*b ^ *c, 13);
    *a += *b; *d = ROTR(*d ^ *a, 9);
    *c += *d; *b = ROTR(*b ^ *c, 5);
}

static void permute(u32 state[16]) {
    for (int round = 0; round < 5; round++) {
        quarter_round(&state[0], &state[4], &state[8], &state[12]);
        quarter_round(&state[1], &state[5], &state[9], &state[13]);
        quarter_round(&state[2], &state[6], &state[10], &state[14]);
        quarter_round(&state[3], &state[7], &state[11], &state[15]);
        quarter_round(&state[0], &state[5], &state[10], &state[15]);
        quarter_round(&state[1], &state[6], &state[11], &state[12]);
        quarter_round(&state[2], &state[7], &state[8], &state[13]);
        quarter_round(&state[3], &state[4], &state[9], &state[14]);
    }
}
#elif defined(__x86_64__)
// Full x86_64 Inline Assembly Implementation (no quarter_round needed)
static void permute(u32 state[16]) {
    asm volatile (
        "push %%rbp\n"
        "mov %%rsp, %%rbp\n"
        "movq %0, %%rsi\n"
        "movl $5, %%ecx\n"
        "1:\n"
        // QR(0,4,8,12)
        "movl (%%rsi), %%eax\n"   "addl 16(%%rsi), %%eax\n" "movl %%eax, (%%rsi)\n"
        "movl 48(%%rsi), %%ebx\n" "xorl %%eax, %%ebx\n"     "rorl $17, %%ebx\n"   "movl %%ebx, 48(%%rsi)\n"
        "movl 32(%%rsi), %%edx\n" "addl %%ebx, %%edx\n"     "movl %%edx, 32(%%rsi)\n"
        "movl 16(%%rsi), %%eax\n" "xorl %%edx, %%eax\n"     "rorl $13, %%eax\n"   "movl %%eax, 16(%%rsi)\n"
        "movl (%%rsi), %%ebx\n"   "addl %%eax, %%ebx\n"     "movl %%ebx, (%%rsi)\n"
        "movl 48(%%rsi), %%edx\n" "xorl %%ebx, %%edx\n"     "rorl $9, %%edx\n"    "movl %%edx, 48(%%rsi)\n"
        "movl 32(%%rsi), %%eax\n" "addl %%edx, %%eax\n"     "movl %%eax, 32(%%rsi)\n"
        "movl 16(%%rsi), %%ebx\n" "xorl %%eax, %%ebx\n"     "rorl $5, %%ebx\n"    "movl %%ebx, 16(%%rsi)\n"
        // QR(1,5,9,13)
        "movl 4(%%rsi), %%eax\n"  "addl 20(%%rsi), %%eax\n" "movl %%eax, 4(%%rsi)\n"
        "movl 52(%%rsi), %%ebx\n" "xorl %%eax, %%ebx\n"     "rorl $17, %%ebx\n"   "movl %%ebx, 52(%%rsi)\n"
        "movl 36(%%rsi), %%edx\n" "addl %%ebx, %%edx\n"     "movl %%edx, 36(%%rsi)\n"
        "movl 20(%%rsi), %%eax\n" "xorl %%edx, %%eax\n"     "rorl $13, %%eax\n"   "movl %%eax, 20(%%rsi)\n"
        "movl 4(%%rsi), %%ebx\n"  "addl %%eax, %%ebx\n"     "movl %%ebx, 4(%%rsi)\n"
        "movl 52(%%rsi), %%edx\n" "xorl %%ebx, %%edx\n"     "rorl $9, %%edx\n"    "movl %%edx, 52(%%rsi)\n"
        "movl 36(%%rsi), %%eax\n" "addl %%edx, %%eax\n"     "movl %%eax, 36(%%rsi)\n"
        "movl 20(%%rsi), %%ebx\n" "xorl %%eax, %%ebx\n"     "rorl $5, %%ebx\n"    "movl %%ebx, 20(%%rsi)\n"
        // QR(2,6,10,14)
        "movl 8(%%rsi), %%eax\n"  "addl 24(%%rsi), %%eax\n" "movl %%eax, 8(%%rsi)\n"
        "movl 56(%%rsi), %%ebx\n" "xorl %%eax, %%ebx\n"     "rorl $17, %%ebx\n"   "movl %%ebx, 56(%%rsi)\n"
        "movl 40(%%rsi), %%edx\n" "addl %%ebx, %%edx\n"     "movl %%edx, 40(%%rsi)\n"
        "movl 24(%%rsi), %%eax\n" "xorl %%edx, %%eax\n"     "rorl $13, %%eax\n"   "movl %%eax, 24(%%rsi)\n"
        "movl 8(%%rsi), %%ebx\n"  "addl %%eax, %%ebx\n"     "movl %%ebx, 8(%%rsi)\n"
        "movl 56(%%rsi), %%edx\n" "xorl %%ebx, %%edx\n"     "rorl $9, %%edx\n"    "movl %%edx, 56(%%rsi)\n"
        "movl 40(%%rsi), %%eax\n" "addl %%edx, %%eax\n"     "movl %%eax, 40(%%rsi)\n"
        "movl 24(%%rsi), %%ebx\n" "xorl %%eax, %%ebx\n"     "rorl $5, %%ebx\n"    "movl %%ebx, 24(%%rsi)\n"
        // QR(3,7,11,15)
        "movl 12(%%rsi), %%eax\n" "addl 28(%%rsi), %%eax\n" "movl %%eax, 12(%%rsi)\n"
        "movl 60(%%rsi), %%ebx\n" "xorl %%eax, %%ebx\n"     "rorl $17, %%ebx\n"   "movl %%ebx, 60(%%rsi)\n"
        "movl 44(%%rsi), %%edx\n" "addl %%ebx, %%edx\n"     "movl %%edx, 44(%%rsi)\n"
        "movl 28(%%rsi), %%eax\n" "xorl %%edx, %%eax\n"     "rorl $13, %%eax\n"   "movl %%eax, 28(%%rsi)\n"
        "movl 12(%%rsi), %%ebx\n" "addl %%eax, %%ebx\n"     "movl %%ebx, 12(%%rsi)\n"
        "movl 60(%%rsi), %%edx\n" "xorl %%ebx, %%edx\n"     "rorl $9, %%edx\n"    "movl %%edx, 60(%%rsi)\n"
        "movl 44(%%rsi), %%eax\n" "addl %%edx, %%eax\n"     "movl %%eax, 44(%%rsi)\n"
        "movl 28(%%rsi), %%ebx\n" "xorl %%eax, %%ebx\n"     "rorl $5, %%ebx\n"    "movl %%ebx, 28(%%rsi)\n"
        // QR(0,5,10,15)
        "movl (%%rsi), %%eax\n"   "addl 20(%%rsi), %%eax\n" "movl %%eax, (%%rsi)\n"
        "movl 60(%%rsi), %%ebx\n" "xorl %%eax, %%ebx\n"     "rorl $17, %%ebx\n"   "movl %%ebx, 60(%%rsi)\n"
        "movl 40(%%rsi), %%edx\n" "addl %%ebx, %%edx\n"     "movl %%edx, 40(%%rsi)\n"
        "movl 20(%%rsi), %%eax\n" "xorl %%edx, %%eax\n"     "rorl $13, %%eax\n"   "movl %%eax, 20(%%rsi)\n"
        "movl (%%rsi), %%ebx\n"   "addl %%eax, %%ebx\n"     "movl %%ebx, (%%rsi)\n"
        "movl 60(%%rsi), %%edx\n" "xorl %%ebx, %%edx\n"     "rorl $9, %%edx\n"    "movl %%edx, 60(%%rsi)\n"
        "movl 40(%%rsi), %%eax\n" "addl %%edx, %%eax\n"     "movl %%eax, 40(%%rsi)\n"
        "movl 20(%%rsi), %%ebx\n" "xorl %%eax, %%ebx\n"     "rorl $5, %%ebx\n"    "movl %%ebx, 20(%%rsi)\n"
        // QR(1,6,11,12)
        "movl 4(%%rsi), %%eax\n"  "addl 24(%%rsi), %%eax\n" "movl %%eax, 4(%%rsi)\n"
        "movl 48(%%rsi), %%ebx\n" "xorl %%eax, %%ebx\n"     "rorl $17, %%ebx\n"   "movl %%ebx, 48(%%rsi)\n"
        "movl 44(%%rsi), %%edx\n" "addl %%ebx, %%edx\n"     "movl %%edx, 44(%%rsi)\n"
        "movl 24(%%rsi), %%eax\n" "xorl %%edx, %%eax\n"     "rorl $13, %%eax\n"   "movl %%eax, 24(%%rsi)\n"
        "movl 4(%%rsi), %%ebx\n"  "addl %%eax, %%ebx\n"     "movl %%ebx, 4(%%rsi)\n"
        "movl 48(%%rsi), %%edx\n" "xorl %%ebx, %%edx\n"     "rorl $9, %%edx\n"    "movl %%edx, 48(%%rsi)\n"
        "movl 44(%%rsi), %%eax\n" "addl %%edx, %%eax\n"     "movl %%eax, 44(%%rsi)\n"
        "movl 24(%%rsi), %%ebx\n" "xorl %%eax, %%ebx\n"     "rorl $5, %%ebx\n"    "movl %%ebx, 24(%%rsi)\n"
        // QR(2,7,8,13)
        "movl 8(%%rsi), %%eax\n"  "addl 28(%%rsi), %%eax\n" "movl %%eax, 8(%%rsi)\n"
        "movl 52(%%rsi), %%ebx\n" "xorl %%eax, %%ebx\n"     "rorl $17, %%ebx\n"   "movl %%ebx, 52(%%rsi)\n"
        "movl 32(%%rsi), %%edx\n" "addl %%ebx, %%edx\n"     "movl %%edx, 32(%%rsi)\n"
        "movl 28(%%rsi), %%eax\n" "xorl %%edx, %%eax\n"     "rorl $13, %%eax\n"   "movl %%eax, 28(%%rsi)\n"
        "movl 8(%%rsi), %%ebx\n"  "addl %%eax, %%ebx\n"     "movl %%ebx, 8(%%rsi)\n"
        "movl 52(%%rsi), %%edx\n" "xorl %%ebx, %%edx\n"     "rorl $9, %%edx\n"    "movl %%edx, 52(%%rsi)\n"
        "movl 32(%%rsi), %%eax\n" "addl %%edx, %%eax\n"     "movl %%eax, 32(%%rsi)\n"
        "movl 28(%%rsi), %%ebx\n" "xorl %%eax, %%ebx\n"     "rorl $5, %%ebx\n"    "movl %%ebx, 28(%%rsi)\n"
        // QR(3,4,9,14)
        "movl 12(%%rsi), %%eax\n" "addl 16(%%rsi), %%eax\n" "movl %%eax, 12(%%rsi)\n"
        "movl 56(%%rsi), %%ebx\n" "xorl %%eax, %%ebx\n"     "rorl $17, %%ebx\n"   "movl %%ebx, 56(%%rsi)\n"
        "movl 36(%%rsi), %%edx\n" "addl %%ebx, %%edx\n"     "movl %%edx, 36(%%rsi)\n"
        "movl 16(%%rsi), %%eax\n" "xorl %%edx, %%eax\n"     "rorl $13, %%eax\n"   "movl %%eax, 16(%%rsi)\n"
        "movl 12(%%rsi), %%ebx\n" "addl %%eax, %%ebx\n"     "movl %%ebx, 12(%%rsi)\n"
        "movl 56(%%rsi), %%edx\n" "xorl %%ebx, %%edx\n"     "rorl $9, %%edx\n"    "movl %%edx, 56(%%rsi)\n"
        "movl 36(%%rsi), %%eax\n" "addl %%edx, %%eax\n"     "movl %%eax, 36(%%rsi)\n"
        "movl 16(%%rsi), %%ebx\n" "xorl %%eax, %%ebx\n"     "rorl $5, %%ebx\n"    "movl %%ebx, 16(%%rsi)\n"
        "decl %%ecx\n"
        "jnz 1b\n"
        "mov %%rbp, %%rsp\n"
        "pop %%rbp\n"
        :
        : "r"(state)
        : "rax", "rbx", "rcx", "rdx", "rsi", "memory"
    );
}
#elif defined(__arm__) || defined(__aarch64__)
// Full ARM Inline Assembly Implementation (no quarter_round needed)
static void permute(u32 state[16]) {
    asm volatile (
        "push {r4-r12, lr}\n"
        "mov r4, %0\n"
        "mov r5, #5\n"
        "1:\n"
        // QR(0,4,8,12)
        "ldr r0, [r4, #0]\n"  "ldr r1, [r4, #16]\n" "add r0, r0, r1\n" "str r0, [r4, #0]\n"
        "ldr r2, [r4, #48]\n" "eor r2, r2, r0\n"   "ror r2, #17\n"   "str r2, [r4, #48]\n"
        "ldr r3, [r4, #32]\n" "add r3, r3, r2\n"   "str r3, [r4, #32]\n"
        "ldr r1, [r4, #16]\n" "eor r1, r1, r3\n"   "ror r1, #13\n"   "str r1, [r4, #16]\n"
        "ldr r0, [r4, #0]\n"  "add r0, r0, r1\n"   "str r0, [r4, #0]\n"
        "ldr r2, [r4, #48]\n" "eor r2, r2, r0\n"   "ror r2, #9\n"    "str r2, [r4, #48]\n"
        "ldr r3, [r4, #32]\n" "add r3, r3, r2\n"   "str r3, [r4, #32]\n"
        "ldr r1, [r4, #16]\n" "eor r1, r1, r3\n"   "ror r1, #5\n"    "str r1, [r4, #16]\n"
        // QR(1,5,9,13)
        "ldr r0, [r4, #4]\n"  "ldr r1, [r4, #20]\n" "add r0, r0, r1\n" "str r0, [r4, #4]\n"
        "ldr r2, [r4, #52]\n" "eor r2, r2, r0\n"   "ror r2, #17\n"   "str r2, [r4, #52]\n"
        "ldr r3, [r4, #36]\n" "add r3, r3, r2\n"   "str r3, [r4, #36]\n"
        "ldr r1, [r4, #20]\n" "eor r1, r1, r3\n"   "ror r1, #13\n"   "str r1, [r4, #20]\n"
        "ldr r0, [r4, #4]\n"  "add r0, r0, r1\n"   "str r0, [r4, #4]\n"
        "ldr r2, [r4, #52]\n" "eor r2, r2, r0\n"   "ror r2, #9\n"    "str r2, [r4, #52]\n"
        "ldr r3, [r4, #36]\n" "add r3, r3, r2\n"   "str r3, [r4, #36]\n"
        "ldr r1, [r4, #20]\n" "eor r1, r1, r3\n"   "ror r1, #5\n"    "str r1, [r4, #20]\n"
        // QR(2,6,10,14)
        "ldr r0, [r4, #8]\n"  "ldr r1, [r4, #24]\n" "add r0, r0, r1\n" "str r0, [r4, #8]\n"
        "ldr r2, [r4, #56]\n" "eor r2, r2, r0\n"   "ror r2, #17\n"   "str r2, [r4, #56]\n"
        "ldr r3, [r4, #40]\n" "add r3, r3, r2\n"   "str r3, [r4, #40]\n"
        "ldr r1, [r4, #24]\n" "eor r1, r1, r3\n"   "ror r1, #13\n"   "str r1, [r4, #24]\n"
        "ldr r0, [r4, #8]\n"  "add r0, r0, r1\n"   "str r0, [r4, #8]\n"
        "ldr r2, [r4, #56]\n" "eor r2, r2, r0\n"   "ror r2, #9\n"    "str r2, [r4, #56]\n"
        "ldr r3, [r4, #40]\n" "add r3, r3, r2\n"   "str r3, [r4, #40]\n"
        "ldr r1, [r4, #24]\n" "eor r1, r1, r3\n"   "ror r1, #5\n"    "str r1, [r4, #24]\n"
        // QR(3,7,11,15)
        "ldr r0, [r4, #12]\n" "ldr r1, [r4, #28]\n" "add r0, r0, r1\n" "str r0, [r4, #12]\n"
        "ldr r2, [r4, #60]\n" "eor r2, r2, r0\n"   "ror r2, #17\n"   "str r2, [r4, #60]\n"
        "ldr r3, [r4, #44]\n" "add r3, r3, r2\n"   "str r3, [r4, #44]\n"
        "ldr r1, [r4, #28]\n" "eor r1, r1, r3\n"   "ror r1, #13\n"   "str r1, [r4, #28]\n"
        "ldr r0, [r4, #12]\n" "add r0, r0, r1\n"   "str r0, [r4, #12]\n"
        "ldr r2, [r4, #60]\n" "eor r2, r2, r0\n"   "ror r2, #9\n"    "str r2, [r4, #60]\n"
        "ldr r3, [r4, #44]\n" "add r3, r3, r2\n"   "str r3, [r4, #44]\n"
        "ldr r1, [r4, #28]\n" "eor r1, r1, r3\n"   "ror r1, #5\n"    "str r1, [r4, #28]\n"
        // QR(0,5,10,15)
        "ldr r0, [r4, #0]\n"  "ldr r1, [r4, #20]\n" "add r0, r0, r1\n" "str r0, [r4, #0]\n"
        "ldr r2, [r4, #60]\n" "eor r2, r2, r0\n"   "ror r2, #17\n"   "str r2, [r4, #60]\n"
        "ldr r3, [r4, #40]\n" "add r3, r3, r2\n"   "str r3, [r4, #40]\n"
        "ldr r1, [r4, #20]\n" "eor r1, r1, r3\n"   "ror r1, #13\n"   "str r1, [r4, #20]\n"
        "ldr r0, [r4, #0]\n"  "add r0, r0, r1\n"   "str r0, [r4, #0]\n"
        "ldr r2, [r4, #60]\n" "eor r2, r2, r0\n"   "ror r2, #9\n"    "str r2, [r4, #60]\n"
        "ldr r3, [r4, #40]\n" "add r3, r3, r2\n"   "str r3, [r4, #40]\n"
        "ldr r1, [r4, #20]\n" "eor r1, r1, r3\n"   "ror r1, #5\n"    "str r1, [r4, #20]\n"
        // QR(1,6,11,12)
        "ldr r0, [r4, #4]\n"  "ldr r1, [r4, #24]\n" "add r0, r0, r1\n" "str r0, [r4, #4]\n"
        "ldr r2, [r4, #48]\n" "eor r2, r2, r0\n"   "ror r2, #17\n"   "str r2, [r4, #48]\n"
        "ldr r3, [r4, #44]\n" "add r3, r3, r2\n"   "str r3, [r4, #44]\n"
        "ldr r1, [r4, #24]\n" "eor r1, r1, r3\n"   "ror r1, #13\n"   "str r1, [r4, #24]\n"
        "ldr r0, [r4, #4]\n"  "add r0, r0, r1\n"   "str r0, [r4, #4]\n"
        "ldr r2, [r4, #48]\n" "eor r2, r2, r0\n"   "ror r2, #9\n"    "str r2, [r4, #48]\n"
        "ldr r3, [r4, #44]\n" "add r3, r3, r2\n"   "str r3, [r4, #44]\n"
        "ldr r1, [r4, #24]\n" "eor r1, r1, r3\n"   "ror r1, #5\n"    "str r1, [r4, #24]\n"
        // QR(2,7,8,13)
        "ldr r0, [r4, #8]\n"  "ldr r1, [r4, #28]\n" "add r0, r0, r1\n" "str r0, [r4, #8]\n"
        "ldr r2, [r4, #52]\n" "eor r2, r2, r0\n"   "ror r2, #17\n"   "str r2, [r4, #52]\n"
        "ldr r3, [r4, #32]\n" "add r3, r3, r2\n"   "str r3, [r4, #32]\n"
        "ldr r1, [r4, #28]\n" "eor r1, r1, r3\n"   "ror r1, #13\n"   "str r1, [r4, #28]\n"
        "ldr r0, [r4, #8]\n"  "add r0, r0, r1\n"   "str r0, [r4, #8]\n"
        "ldr r2, [r4, #52]\n" "eor r2, r2, r0\n"   "ror r2, #9\n"    "str r2, [r4, #52]\n"
        "ldr r3, [r4, #32]\n" "add r3, r3, r2\n"   "str r3, [r4, #32]\n"
        "ldr r1, [r4, #28]\n" "eor r1, r1, r3\n"   "ror r1, #5\n"    "str r1, [r4, #28]\n"
        // QR(3,4,9,14)
        "ldr r0, [r4, #12]\n" "ldr r1, [r4, #16]\n" "add r0, r0, r1\n" "str r0, [r4, #12]\n"
        "ldr r2, [r4, #56]\n" "eor r2, r2, r0\n"   "ror r2, #17\n"   "str r2, [r4, #56]\n"
        "ldr r3, [r4, #36]\n" "add r3, r3, r2\n"   "str r3, [r4, #36]\n"
        "ldr r1, [r4, #16]\n" "eor r1, r1, r3\n"   "ror r1, #13\n"   "str r1, [r4, #16]\n"
        "ldr r0, [r4, #12]\n" "add r0, r0, r1\n"   "str r0, [r4, #12]\n"
        "ldr r2, [r4, #56]\n" "eor r2, r2, r0\n"   "ror r2, #9\n"    "str r2, [r4, #56]\n"
        "ldr r3, [r4, #36]\n" "add r3, r3, r2\n"   "str r3, [r4, #36]\n"
        "ldr r1, [r4, #16]\n" "eor r1, r1, r3\n"   "ror r1, #5\n"    "str r1, [r4, #16]\n"
        "subs r5, r5, #1\n"
        "bne 1b\n"
        "pop {r4-r12, lr}\n"
        :
        : "r"(state)
        : "r0", "r1", "r2", "r3", "r4", "r5", "memory"
    );
}
#else
// Portable C Implementation
static inline void quarter_round(u32 *a, u32 *b, u32 *c, u32 *d) {
    *a += *b; *d = ROTR(*d ^ *a, 17);
    *c += *d; *b = ROTR(*b ^ *c, 13);
    *a += *b; *d = ROTR(*d ^ *a, 9);
    *c += *d; *b = ROTR(*b ^ *c, 5);
}

static void permute(u32 state[16]) {
    for (int round = 0; round < 5; round++) {
        quarter_round(&state[0], &state[4], &state[8], &state[12]);
        quarter_round(&state[1], &state[5], &state[9], &state[13]);
        quarter_round(&state[2], &state[6], &state[10], &state[14]);
        quarter_round(&state[3], &state[7], &state[11], &state[15]);
        quarter_round(&state[0], &state[5], &state[10], &state[15]);
        quarter_round(&state[1], &state[6], &state[11], &state[12]);
        quarter_round(&state[2], &state[7], &state[8], &state[13]);
        quarter_round(&state[3], &state[4], &state[9], &state[14]);
    }
}
#endif

// Compression Function
static void cpuhash256_compress(u32 state[16], const u32 block[16]) {
    u32 orig_state[16];
    memcpy(orig_state, state, 64);
    for (int i = 0; i < 16; i++) state[i] ^= block[i];
    permute(state);
    for (int i = 0; i < 16; i++) state[i] ^= orig_state[i];
}

void cpuhash256(const u8 *input, size_t len, u8 output[32]) {
    u32 state[16];
    memcpy(state, IV, 64);
    size_t pos = 0;

    while (pos + 64 <= len) {
        u32 block[16];
#if defined(__AVX2__)
        load_block_avx2(input + pos, block);
#elif defined(__ARM_NEON)
        load_block_neon(input + pos, block);
#else
        for (int i = 0; i < 16; i++) {
            block[i] = ((u32)input[pos + 4*i] << 24) |
                       ((u32)input[pos + 4*i+1] << 16) |
                       ((u32)input[pos + 4*i+2] << 8) |
                       input[pos + 4*i+3];
        }
#endif
        cpuhash256_compress(state, block);
        pos += 64;
    }

    u8 last_block[64] = {0};
    size_t remaining = len - pos;
    memcpy(last_block, input + pos, remaining);
    last_block[remaining] = 0x80;
    if (remaining >= 56) {
        u32 block[16];
#if defined(__AVX2__)
        load_block_avx2(last_block, block);
#elif defined(__ARM_NEON)
        load_block_neon(last_block, block);
#else
        for (int i = 0; i < 16; i++) {
            block[i] = ((u32)last_block[4*i] << 24) |
                       ((u32)last_block[4*i+1] << 16) |
                       ((u32)last_block[4*i+2] << 8) |
                       last_block[4*i+3];
        }
#endif
        cpuhash256_compress(state, block);
        memset(last_block, 0, 64);
    }
    u64 bits = len * 8;
    for (int i = 0; i < 8; i++) {
        last_block[56 + i] = (bits >> (56 - 8*i)) & 0xFF;
    }
    u32 last_block_u32[16];
#if defined(__AVX2__)
    load_block_avx2(last_block, last_block_u32);
#elif defined(__ARM_NEON)
    load_block_neon(last_block, last_block_u32);
#else
    for (int i = 0; i < 16; i++) {
        last_block_u32[i] = ((u32)last_block[4*i] << 24) |
                            ((u32)last_block[4*i+1] << 16) |
                            ((u32)last_block[4*i+2] << 8) |
                            last_block[4*i+3];
    }
#endif
    cpuhash256_compress(state, last_block_u32);

    for (int i = 0; i < 8; i++) {
        output[4*i] = (state[i] >> 24) & 0xFF;
        output[4*i+1] = (state[i] >> 16) & 0xFF;
        output[4*i+2] = (state[i] >> 8) & 0xFF;
        output[4*i+3] = state[i] & 0xFF;
    }
}