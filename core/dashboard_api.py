from fastapi import FastAPI
from typing import Dict, List
import time

app = FastAPI(title="Digital Evolution Dashboard API")

# Global state to be populated by the main bot loop
bot_state = {
    "open_trades": [],
    "balance": 0.0,
    "last_decisions": [],
    "heatmap_data": []
}

@app.get("/status")
async def get_status():
    """
    Phase 56: Returns current open trades, balance, and last 5 ML decisions.
    """
    return {
        "timestamp": time.time(),
        "balance": bot_state["balance"],
        "open_trades": bot_state["open_trades"],
        "last_5_decisions": bot_state["last_decisions"][-5:]
    }

@app.get("/heatmap")
async def get_heatmap():
    """
    Phase 57: Returns current Orderbook Walls and OFI data for all scanned symbols.
    """
    return {
        "timestamp": time.time(),
        "heatmap": bot_state["heatmap_data"]
    }

def update_bot_state(open_trades: List, balance: float, last_decisions: List, scan_results: List):
    """
    Updates the global state with fresh data from the bot.
    """
    bot_state["open_trades"] = open_trades
    bot_state["balance"] = balance
    bot_state["last_decisions"] = last_decisions
    
    # Process scan results for heatmap
    heatmap = []
    for res in scan_results:
        heatmap.append({
            "symbol": res["symbol"],
            "wall_dist": res["wall_dist"],
            "ofi": res["ofi"],
            "cvd": res["cvd"],
            "sentiment": res["sentiment_proxy"]
        })
    bot_state["heatmap_data"] = heatmap
