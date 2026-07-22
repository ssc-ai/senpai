"""Process memory reclamation helpers."""

import ctypes
import ctypes.util
import gc
import logging

logger = logging.getLogger(__name__)


def reclaim_process_memory() -> None:
    """Return freed heap to the OS after a detection run.

    Large transient numpy arrays (e.g. the upsampled ``rectangle_pyramoid`` working arrays)
    are freed promptly, but glibc keeps the freed pages in its arena free-lists instead of
    returning them to the OS, so RSS ratchets up across requests in long-lived workers.
    Collect Python cycles, then ask glibc to trim its arenas. LRU caches are left intact.
    No-op on non-glibc platforms.
    """
    gc.collect()
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError) as exc:  # non-glibc / malloc_trim absent
        logger.debug("malloc_trim unavailable; skipping: %s", exc)
