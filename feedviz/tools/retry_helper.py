# feedviz/tools/retry_helper.py

import time
import functools


def retry_with_backoff(max_retries: int = 3, base_delay: float = 30.0):
    """
    Decorator that retries a function with exponential backoff
    when rate limit errors occur.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e).lower()
                    if "429" in error_msg or "rate limit" in error_msg:
                        wait_time = base_delay * (2 ** attempt)
                        print(f"[RateLimit] Attempt {attempt + 1}/{max_retries}. "
                              f"Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        raise e
            return func(*args, **kwargs)
        return wrapper
    return decorator