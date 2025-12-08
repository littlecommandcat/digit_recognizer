import asyncio
import time
from functools import wraps
from .errors import *

ERROR_MAP = {
    "save": ModelSaveError,
    "load": ModelLoadError,
    "file": ModelFileNotFoundError,
    "image": ImageProcessingError,
    "format": InvalidImageFormatError,
    "predict": PredictionError,
    "template": NoTemplatesError,
}

def RunTimeChecker(func):
    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            _starttime = time.time()
            elapsed = time.time() - _starttime
            if getattr(self, "timeout", None) is not None and elapsed >= self.timeout and self.timeout > 0:
                raise TimeoutError(f"{func.__name__} Timeout reached")
            return await func(self, *args, **kwargs)
        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            _starttime = time.time()
            elapsed = time.time() - _starttime
            if getattr(self, "timeout", None) is not None and elapsed >= self.timeout and self.timeout > 0:
                raise TimeoutError(f"{func.__name__} Timeout reached")
            return func(self, *args, **kwargs)
        return sync_wrapper

def ErrorCatcher(func):
    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            if not hasattr(self, "_error_catcher") or self._error_catcher is False:
                return await func(self, *args, **kwargs)
            try:
                return await func(self, *args, **kwargs)
            except ModelFileNotFoundError as e:
                print(f"[ModelFileNotFoundError] {e}")

            except ModelSaveError as e:
                print(f"[ModelSaveError] {e}")

            except ModelLoadError as e:
                print(f"[ModelLoadError] {e}")

            except ImageProcessingError as e:
                print(f"[ImageProcessingError] {e}")

            except InvalidImageFormatError as e:
                print(f"[InvalidImageFormatError] {e}")

            except FeatureError as e:
                print(f"[FeatureError] {e}")

            except NoTemplatesError as e:
                print(f"[NoTemplatesError] {e}")

            except PredictionError as e:
                print(f"[PredictionError] {e}")

            except DigitRecognizeError as e:
                print(f"[DigitRecognizeError] {e}")

            except Exception as e:
                pass
                # Already handle
                # print(f"[Unknown Error] {e}")
        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            if not hasattr(self, "_error_catcher") or self._error_catcher is False:
                return func(self, *args, **kwargs)
            try:
                return func(self, *args, **kwargs)
            except ModelFileNotFoundError as e:
                print(f"[ModelFileNotFoundError] {e}")

            except ModelSaveError as e:
                print(f"[ModelSaveError] {e}")

            except ModelLoadError as e:
                print(f"[ModelLoadError] {e}")

            except ImageProcessingError as e:
                print(f"[ImageProcessingError] {e}")

            except InvalidImageFormatError as e:
                print(f"[InvalidImageFormatError] {e}")

            except FeatureError as e:
                print(f"[FeatureError] {e}")

            except NoTemplatesError as e:
                print(f"[NoTemplatesError] {e}")

            except PredictionError as e:
                print(f"[PredictionError] {e}")

            except DigitRecognizeError as e:
                print(f"[DigitRecognizeError] {e}")

            except Exception as e:
                pass
                # Already handle
                # print(f"[Unknown Error] {e}")
        return sync_wrapper
