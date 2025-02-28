try:
    from .cpuhash256_wrapper import hash256
    __all__ = ['hash256']
except ImportError as e:
    raise ImportError(f"Failed to import cpuhash256 module: {e}. Make sure the module is properly installed.")
