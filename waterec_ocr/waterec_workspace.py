"""Per-recognizer scratch ownership for sequential TensorRT execution contexts.

All borrowers must execute on one CUDA stream and must bind the lease before
CUDA graph capture. Tensor/output buffers are not workspace and stay separate.
A pool cannot grow after its first allocation: captured pointers must be stable.
The caller supplies the allocator so this module has no CUDA import side effects.
"""


class ScratchPool:
    def __init__(self, allocate):
        self._allocate = allocate
        self._buffer = None
        self._capacity = 0
        self._stream = None
        self._users = 0
        self._closed = False

    @property
    def capacity(self):
        return self._capacity

    @property
    def active_leases(self):
        return self._users

    @property
    def closed(self):
        return self._closed

    def acquire(self, nbytes, stream_handle):
        if self._closed:
            raise RuntimeError("Scratch pool is closed")
        size = int(nbytes)
        if size < 0 or size != nbytes:
            raise ValueError("Workspace size must be a nonnegative integer")
        stream = int(stream_handle)
        if self._buffer is not None:
            if stream != self._stream:
                raise ValueError("Shared workspace requires the same CUDA stream")
            if size > self._capacity:
                raise ValueError("Workspace cannot grow after pointers are bound")
        else:
            buffer = self._allocate(max(1, size))
            self._buffer = buffer
            self._capacity = max(1, size)
            self._stream = stream
        self._users += 1
        return _ScratchLease(self)

    def _release(self):
        if self._users <= 0:
            raise RuntimeError("Scratch pool has no active lease")
        if self._users == 1:
            # Leave ownership intact if the backend reports a failed free.
            self._buffer.free()
            self._buffer = None
            self._closed = True
        self._users -= 1

    def close(self):
        if self._users:
            raise RuntimeError("Close execution contexts before their scratch pool")
        self._closed = True


class _ScratchLease:
    def __init__(self, pool):
        self._pool = pool
        self._closed = False

    def __int__(self):
        if self._closed or self._pool.closed:
            raise RuntimeError("Workspace lease is closed")
        return int(self._pool._buffer)

    def free(self):
        if self._closed:
            return
        self._pool._release()
        self._closed = True
