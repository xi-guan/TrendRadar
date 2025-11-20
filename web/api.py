from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from typing import Optional, List
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_server.tools.data_query import DataQueryTools
from mcp_server.tools.analytics import AnalyticsTools
from mcp_server.tools.search_tools import SearchTools

app = FastAPI(title="TrendRadar Web")

project_root = str(Path(__file__).parent.parent)
data_tools = DataQueryTools(project_root)
analytics_tools = AnalyticsTools(project_root)
search_tools = SearchTools(project_root)

def load_config():
    config_path = Path(project_root) / "config" / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

@app.get("/api/news/latest")
async def get_latest_news(
    platforms: Optional[str] = Query(None),
    limit: int = Query(50, le=1000),
    include_url: bool = Query(False)
):
    platform_list = platforms.split(",") if platforms else None
    return data_tools.get_latest_news(platform_list, limit, include_url)

@app.get("/api/topics/trending")
async def get_trending_topics(top_n: int = Query(10, le=100)):
    return analytics_tools.get_trending_topics(top_n)

@app.get("/api/search")
async def search_news(
    keyword: str = Query(...),
    platforms: Optional[str] = Query(None),
    limit: int = Query(50, le=1000)
):
    platform_list = platforms.split(",") if platforms else None
    return search_tools.search_news(keyword, platform_list, limit)

@app.get("/api/stats")
async def get_stats():
    result = data_tools.get_latest_news(limit=10000)
    total = result.get('total', 0)
    config = load_config()
    total_platforms = len(config.get('platforms', []))

    latest_timestamp = "N/A"
    if result.get('news'):
        latest_timestamp = result['news'][0].get('timestamp', 'N/A')

    return {
        "total_news": total,
        "total_platforms": total_platforms,
        "latest_timestamp": latest_timestamp
    }

@app.get("/api/platforms")
async def get_platforms():
    config = load_config()
    platforms = config.get('platforms', [])
    return {"platforms": platforms}

app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"), html=True), name="static")
