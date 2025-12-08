import asyncio
import time
from functools import wraps
from .errors import *

def RunTimeChecker(func):
    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            if not hasattr(self, "_starttime") or self._starttime is None:
                self._starttime = time.time()
            elapsed = time.time() - self._starttime
            if getattr(self, "timeout", None) is not None and elapsed >= self.timeout and self.timeout > 0:
                raise TimeoutError(f"{func.__name__} Timeout reached")
            return await func(self, *args, **kwargs)
        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            if not hasattr(self, "_starttime") or self._starttime is None:
                self._starttime = time.time()
            elapsed = time.time() - self._starttime
            if getattr(self, "timeout", None) is not None and elapsed >= self.timeout and self.timeout > 0:
                raise TimeoutError(f"{func.__name__} Timeout reached")
            return func(self, *args, **kwargs)
        return sync_wrapper

def ErrorCatcher(func):
    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            try:
                return await func(self, *args, **kwargs)
            except FeatureError as e:
                print(f"FeatureError caught in {func.__name__}: {e}")
            except ModalError as e:
                print(f"ModalError caught in {func.__name__}: {e}")
            except Exception as e:
                print(f"Unhandled exception in {func.__name__}: {e}")
        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except FeatureError as e:
                print(f"FeatureError caught in {func.__name__}: {e}")
            except ModalError as e:
                print(f"ModalError caught in {func.__name__}: {e}")
            except Exception as e:
                print(f"Unhandled exception in {func.__name__}: {e}")
        return sync_wrapper
