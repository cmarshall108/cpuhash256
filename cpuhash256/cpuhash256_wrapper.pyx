from libc.stdint cimport uint8_t
from libc.stddef cimport size_t
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING
from cpython.buffer cimport PyBUF_SIMPLE, PyObject_GetBuffer, PyBuffer_Release, Py_buffer

cdef extern from "cpuhash256.h":
    void cpuhash256(const uint8_t* message, size_t len, uint8_t hash[32]) nogil

def hash256(data):
    """Optimized CPU hash computation"""
    cdef:
        Py_buffer view
        uint8_t[32] hash_out
        const uint8_t* buf_ptr
        size_t length
        bytes result
        
    # Get direct buffer access to input data
    try:
        PyObject_GetBuffer(data, &view, PyBUF_SIMPLE)
        buf_ptr = <const uint8_t*>view.buf
        length = view.len
        
        # Release the GIL during hash computation
        with nogil:
            cpuhash256(buf_ptr, length, hash_out)
            
        # Create bytes object directly from hash_out
        result = PyBytes_FromStringAndSize(<char*>hash_out, 32)
        return result
    finally:
        PyBuffer_Release(&view)

__all__ = ['hash256']