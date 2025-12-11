"""
E2B Sandbox Cache Manager

Manages persistent E2B sandbox instances per thread_id with automatic cleanup.
"""

import time
from typing import Dict
from loguru import logger

try:
    from e2b_code_interpreter import Sandbox

    E2B_AVAILABLE = True
except ImportError:
    E2B_AVAILABLE = False
    Sandbox = None

from cuga.config import settings


class SandboxCacheEntry:
    """Entry in the sandbox cache containing sandbox instance and metadata."""

    def __init__(self, sandbox: "Sandbox", thread_id: str):
        self.sandbox = sandbox
        self.thread_id = thread_id
        self.created_at = time.time()
        self.last_used = time.time()
        self.use_count = 0

    def mark_used(self):
        """Update last used timestamp and increment use count."""
        self.last_used = time.time()
        self.use_count += 1

    def is_expired(self, ttl_seconds: int = 3600) -> bool:
        """Check if entry has expired based on TTL (default 1 hour)."""
        age = time.time() - self.created_at
        return age > ttl_seconds

    def is_alive(self) -> bool:
        """Check if sandbox is still alive and responsive."""
        try:
            # Simple health check - try to execute trivial code
            execution = self.sandbox.run_code("print('alive')")
            return execution.error is None
        except Exception as e:
            logger.debug(f"Sandbox health check failed: {e}")
            return False

    def get_age(self) -> float:
        """Get age in seconds since creation."""
        return time.time() - self.created_at

    def get_idle_time(self) -> float:
        """Get idle time in seconds since last use."""
        return time.time() - self.last_used


class E2BSandboxCache:
    """
    Cache manager for E2B sandbox instances.

    Maintains one sandbox per thread_id with automatic cleanup of expired/dead sandboxes.
    """

    _instance = None
    _sandboxes: Dict[str, SandboxCacheEntry] = {}
    _ttl_seconds: int = 3600  # 1 hour default
    _template_name: str = "cuga-langchain"

    def __new__(cls):
        """Singleton pattern to ensure one cache instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the cache."""
        if not hasattr(self, "_initialized"):
            self._sandboxes = {}
            self._ttl_seconds = 3600
            self._template_name = "cuga-langchain"
            self._initialized = True

    def set_ttl(self, seconds: int):
        """Set time-to-live for cached sandboxes."""
        self._ttl_seconds = seconds
        logger.info(f"E2B sandbox TTL set to {seconds} seconds ({seconds / 3600:.1f} hours)")

    def set_template(self, template_name: str):
        """Set E2B template name for sandbox creation."""
        self._template_name = template_name
        logger.info(f"E2B sandbox template set to: {template_name}")

    def get_or_create(self, thread_id: str) -> "Sandbox":
        """
        Get existing sandbox for thread_id or create new one.

        Args:
            thread_id: Unique identifier for the conversation thread

        Returns:
            E2B Sandbox instance

        Raises:
            RuntimeError: If E2B is not available or sandbox creation fails
        """
        if not E2B_AVAILABLE:
            raise RuntimeError("e2b-code-interpreter package not installed")

        sandbox_mode = settings.advanced_features.e2b_sandbox_mode

        # If mode is "single" or "per-call", create new instance without caching
        if sandbox_mode in ("single", "per-call"):
            logger.info(
                f"Creating new E2B sandbox (mode: {sandbox_mode}) for thread {thread_id} "
                f"with template '{self._template_name}'"
            )
            try:
                sandbox = Sandbox.create(self._template_name)
                logger.info(f"Successfully created sandbox for thread {thread_id} (mode: {sandbox_mode})")
                return sandbox
            except Exception as e:
                logger.error(f"Failed to create E2B sandbox for thread {thread_id}: {e}")
                raise RuntimeError(f"Failed to create E2B sandbox: {e}") from e

        # Per-session mode: use caching logic
        # Clean up expired/dead sandboxes before proceeding
        self._cleanup_expired()

        # Check if we have a valid cached sandbox
        if thread_id in self._sandboxes:
            entry = self._sandboxes[thread_id]

            # Check if expired
            if entry.is_expired(self._ttl_seconds):
                logger.info(
                    f"Sandbox for thread {thread_id} expired (age: {entry.get_age():.1f}s), creating new one"
                )
                self._remove_sandbox(thread_id)
            # Check if alive
            elif not entry.is_alive():
                logger.warning(
                    f"Sandbox for thread {thread_id} is dead (age: {entry.get_age():.1f}s), creating new one"
                )
                self._remove_sandbox(thread_id)
            else:
                # Valid cached sandbox found
                entry.mark_used()
                logger.info(
                    f"Reusing cached sandbox for thread {thread_id} "
                    f"(age: {entry.get_age():.1f}s, uses: {entry.use_count}, idle: {entry.get_idle_time():.1f}s)"
                )
                return entry.sandbox

        # Create new sandbox
        logger.info(f"Creating new E2B sandbox for thread {thread_id} with template '{self._template_name}'")
        try:
            sandbox = Sandbox.create(self._template_name)
            entry = SandboxCacheEntry(sandbox, thread_id)
            entry.mark_used()
            self._sandboxes[thread_id] = entry
            logger.info(
                f"Successfully created sandbox for thread {thread_id} (total cached: {len(self._sandboxes)})"
            )
            return sandbox
        except Exception as e:
            logger.error(f"Failed to create E2B sandbox for thread {thread_id}: {e}")
            raise RuntimeError(f"Failed to create E2B sandbox: {e}") from e

    def _remove_sandbox(self, thread_id: str):
        """Remove and cleanup sandbox for given thread_id."""
        if thread_id in self._sandboxes:
            entry = self._sandboxes[thread_id]
            try:
                entry.sandbox.kill()
                logger.debug(f"Killed sandbox for thread {thread_id}")
            except Exception as e:
                logger.debug(f"Error killing sandbox for thread {thread_id}: {e}")
            finally:
                del self._sandboxes[thread_id]

    def _cleanup_expired(self):
        """Remove expired and dead sandboxes from cache."""
        threads_to_remove = []

        for thread_id, entry in self._sandboxes.items():
            if entry.is_expired(self._ttl_seconds):
                logger.info(
                    f"Cleaning up expired sandbox for thread {thread_id} "
                    f"(age: {entry.get_age():.1f}s, uses: {entry.use_count})"
                )
                threads_to_remove.append(thread_id)
            elif not entry.is_alive():
                logger.warning(
                    f"Cleaning up dead sandbox for thread {thread_id} "
                    f"(age: {entry.get_age():.1f}s, uses: {entry.use_count})"
                )
                threads_to_remove.append(thread_id)

        for thread_id in threads_to_remove:
            self._remove_sandbox(thread_id)

    def remove(self, thread_id: str):
        """Manually remove sandbox for specific thread_id."""
        if thread_id in self._sandboxes:
            logger.info(f"Manually removing sandbox for thread {thread_id}")
            self._remove_sandbox(thread_id)

    def clear_all(self):
        """Clear all cached sandboxes."""
        logger.info(f"Clearing all cached sandboxes ({len(self._sandboxes)} total)")
        threads = list(self._sandboxes.keys())
        for thread_id in threads:
            self._remove_sandbox(thread_id)

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        stats = {
            "total_sandboxes": len(self._sandboxes),
            "ttl_seconds": self._ttl_seconds,
            "template_name": self._template_name,
            "sandboxes": {},
        }

        for thread_id, entry in self._sandboxes.items():
            stats["sandboxes"][thread_id] = {
                "age_seconds": entry.get_age(),
                "idle_seconds": entry.get_idle_time(),
                "use_count": entry.use_count,
                "is_alive": entry.is_alive(),
            }

        return stats


# Global cache instance
_sandbox_cache = E2BSandboxCache()


def get_sandbox_cache() -> E2BSandboxCache:
    """Get the global sandbox cache instance."""
    return _sandbox_cache
