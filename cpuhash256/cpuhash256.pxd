from libc.stdint cimport uint8_t
from libc.stddef cimport size_t

cdef extern from "cpuhash256.h":
    void cpuhash256(const uint8_t* message, size_t len, uint8_t hash[32])
