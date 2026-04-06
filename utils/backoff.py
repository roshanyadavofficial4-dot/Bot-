import asyncio
import logging
import functools

logger = logging.getLogger("Backoff")

def exponential_backoff(retries: int = 5, base_delay: float = 1.0):
    """
    Decorator for exponential backoff on exchange calls (Killer #7).
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Don't retry on certain errors like InsufficientFunds or InvalidOrder
                    import ccxt
                    if isinstance(e, (ccxt.InsufficientFunds, ccxt.InvalidOrder, ccxt.AuthenticationError)):
                        raise e
                        
                    if i == retries - 1:
                        logger.error(f"Backoff: {func.__name__} failed after {retries} attempts ({e}).")
                        raise e
                    
                    delay = base_delay * (2 ** i)
                    logger.warning(f"Backoff: {func.__name__} failed ({e}). Retrying in {delay:.2f}s (Attempt {i+1}/{retries})...")
                    await asyncio.sleep(delay)
        return wrapper
    return decorator
