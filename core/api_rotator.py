import time
import logging
from config import GEMINI_API_KEYS

# Configure module-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("GeminiAPIRotator")

class GeminiAPIRotator:
    """
    Robust Round-Robin API Key Rotator with Rate-Limit Bypassing.
    Ensures high-frequency trading loops are never broken by 429 errors.
    """
    
    def __init__(self, cooldown_seconds: int = 60):
        self.keys = GEMINI_API_KEYS
        self.num_keys = len(self.keys)
        self.current_index = 0
        self.cooldown_seconds = cooldown_seconds
        
        # Tracks the exact Unix timestamp when a key becomes available again
        self.cooldowns = {index: 0.0 for index in range(self.num_keys)}
        
        if self.num_keys == 0:
            logger.error("CRITICAL: No API keys loaded. Engine cannot start.")
            raise ValueError("API keys list cannot be empty.")

    def get_active_key(self) -> str:
        """
        Iterates sequentially through keys. Instantly skips any key currently 
        flagged with a cooldown timestamp, ensuring the main loop never blocks.
        """
        start_index = self.current_index
        
        while True:
            current_time = time.time()
            
            # Check if the current key has passed its cooldown period
            if current_time >= self.cooldowns[self.current_index]:
                active_key = self.keys[self.current_index]
                logger.info(f"Switching to Key index {self.current_index}")
                
                # Advance index for the next call (Round-Robin)
                self.current_index = (self.current_index + 1) % self.num_keys
                return active_key
            
            # Key is on cooldown, move to the next one
            self.current_index = (self.current_index + 1) % self.num_keys
            
            # Failsafe: If ALL 10 keys are exhausted/rate-limited simultaneously
            if self.current_index == start_index:
                logger.warning("All keys are currently rate-limited. Throttling for 1s to prevent CPU thrashing...")
                time.sleep(1)

    def flag_rate_limit(self, failed_key: str):
        """
        Triggered by the Try/Except block in the API caller when a 429 or timeout occurs.
        Applies a strict cooldown to the failing key.
        """
        try:
            key_index = self.keys.index(failed_key)
            self.cooldowns[key_index] = time.time() + self.cooldown_seconds
            logger.warning(f"Key {key_index} rate-limited, cooling down for {self.cooldown_seconds}s")
        except ValueError:
            logger.error("Attempted to flag an unknown API key.")
