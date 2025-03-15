# main.py
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import httpx
import json
import os
from bs4 import BeautifulSoup 
from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
from mcp.server import Server
import uvicorn

load_dotenv()

mcp = FastMCP("Agentdocs")

USER_AGENT = "Agentdocs-app/1.0"
SERPER_URL="https://google.serper.dev/search"

docs_urls = {
"langchain": "python.langchain.com/docs",
"llama-index": "docs.llamaindex.ai/en/stable",
"autogen":"microsoft.github.io/autogen/stable",
"agno":"docs.agno.com",
"openai-agents-sdk": "openai.github.io/openai-agents-python",
"mcp-doc":"modelcontextprotocol.io",
"camel-ai":"docs.camel-ai.org",
"crew-ai":"docs.crewai.com"
}

async def search_web(query: str) -> dict | None:
    payload = json.dumps({"q": query, "num": 2})

    headers = {
        "X-API-KEY": os.getenv("SERPER_API_KEY"),
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                SERPER_URL, headers=headers, data=payload, timeout=30.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            return {"organic": []}
  
async def fetch_url(url: str):
  async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=30.0)
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()
            return text
        except httpx.TimeoutException:
            return "Timeout error"

@mcp.tool()  
async def get_docs(query: str, library: str):
  """
    搜索给定查询和库的最新文档。
    支持 langchain、llama-index、autogen、agno、openai-agents-sdk、mcp-doc、camel-ai 和 crew-ai。

    参数:
    query: 要搜索的查询 (例如 "React Agent")
    library: 要搜索的库 (例如 "agno")

    返回:
    文档中的文本
  """
  if library not in docs_urls:
    raise ValueError(f"Library {library} not supported by this tool")
  
  query = f"site:{docs_urls[library]} {query}"
  results = await search_web(query)
  if len(results["organic"]) == 0:
    return "No results found"
  
  text = ""
  for result in results["organic"]:
    text += await fetch_url(result["link"])
 
  return text


# Stdio协议
if __name__ == "__main__":
    mcp.run(transport="stdio")
 

#  # SSE协议
# def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
#     """Create a Starlette application that can server the provied mcp server with SSE."""
#     sse = SseServerTransport("/messages/")

#     async def handle_sse(request: Request) -> None:
#         async with sse.connect_sse(
#                 request.scope,
#                 request.receive,
#                 request._send,  # noqa: SLF001
#         ) as (read_stream, write_stream):
#             await mcp_server.run(
#                 read_stream,
#                 write_stream,
#                 mcp_server.create_initialization_options(),
#             )

#     return Starlette(
#         debug=debug,
#         routes=[
#             Route("/sse", endpoint=handle_sse),
#             Mount("/messages/", app=sse.handle_post_message),
#         ],
#     )

# if __name__ == "__main__":
#     mcp_server = mcp._mcp_server   

#     import argparse
    
#     parser = argparse.ArgumentParser(description='Run MCP SSE-based server')
#     parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
#     parser.add_argument('--port', type=int, default=8020, help='Port to listen on')
#     args = parser.parse_args()

#     # Bind SSE request handling to MCP server
#     starlette_app = create_starlette_app(mcp_server, debug=True)

#     uvicorn.run(starlette_app, host=args.host, port=args.port)
