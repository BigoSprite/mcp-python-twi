import asyncio
import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from .twi_server_types import *
import logging

# 配置loggging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - S - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
# 获取logger实例
logger = logging.getLogger(__name__)

SERVER_NAME = "TWI Server"

# Create a server instance
server = Server(SERVER_NAME)

# 创建全局工具处理器实例
_tool_handler = ToolHandler()


@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """列出可用工具"""
    return ToolDefinition.get_tool_definitions()


@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> types.CallToolResult:
    """处理工具调用请求"""
    return await _tool_handler.handle_tool_call(name, arguments)


@server.list_prompts()
async def handle_list_prompts() -> List[types.Prompt]:
    """列出可用提示词"""
    return [
        types.Prompt(
            name=TWIPrompts.TEST_ECHO_PROMPT,
            title="Test Echo Prompt",
            description="A simple prompt tha can echo text",
            arguments=[
                types.PromptArgument(
                    name="context",
                    description="echo_prompt parameter",
                    required=True,
                )
            ],
        )

    ]


async def run():
    """Run the server with lifespan management."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=SERVER_NAME,
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main():
    """Entry point for the client script."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
