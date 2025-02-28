#ifndef CPUHASH256_H
#define CPUHASH256_H

#include <stdint.h>
#include <stddef.h>

void cpuhash256(const uint8_t* message, size_t len, uint8_t hash[32]);

#endif