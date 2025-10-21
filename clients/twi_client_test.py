"""
cd to the `python` directory and run:
    uv run twi-client-dev
"""

import asyncio
import os
import json
from sys import path
import time
import statistics
import base64
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable, Awaitable, Optional

from PIL import Image as PILImage
import io
import logging

from pydantic import AnyUrl, BaseModel, Field

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.shared.context import RequestContext

# 配置loggging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - C - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
# 获取logger实例
logger = logging.getLogger(__name__)


# ========== 配置和常量定义 ==========
class ClientConfig:
    """客户端配置"""
    SERVER_COMMAND = "uv"
    SERVER_ARGS = ["run", "mcp-server-twi"]
    TEST_ITERATIONS = 50
    PROGRESS_INTERVAL = 10
    TEST_IMAGE_DIR = "./test_data"


# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command=ClientConfig.SERVER_COMMAND,
    args=ClientConfig.SERVER_ARGS,
    env={"UV_INDEX": os.environ.get("UV_INDEX", "")},
)


# ========== 数据模型定义 ==========
class PerformanceStats(BaseModel):
    """性能统计结果"""
    average_time: float = Field(description="平均耗时")
    max_time: float = Field(description="最大耗时")
    min_time: float = Field(description="最小耗时")
    total_time: float = Field(description="总耗时")
    call_count: int = Field(description="调用次数")
    std_dev: float = Field(description="标准差")
    stability: str = Field(description="稳定性评估")


class ZoomInImageParams(BaseModel):
    """缩放图像参数"""
    image_path: str = Field(description="图像路径")
    shape_n: float = Field(default=1, description="批次数量")
    shape_c: float = Field(default=3, description="通道数")
    shape_w: int = Field(description="图像宽度")
    shape_h: int = Field(description="图像高度")
    coord_x1: int = Field(description="左上角x坐标")
    coord_y1: int = Field(description="左上角y坐标")
    coord_x2: int = Field(description="右下角x坐标")
    coord_y2: int = Field(description="右下角y坐标")
    image_format: str = Field(default="jpg", description="图像格式")
    config: Optional[Dict[str, Any]] = Field(default=None, description="配置参数")
    tag: str = Field(default="NotSetTag", description="调试标签")
    dbg_dump: bool = Field(default=True, description="调试模式")


class TestCase(BaseModel):
    """测试用例"""
    name: str = Field(description="测试用例名称")
    text: str = Field(description="测试文本")
    expected_success: bool = Field(description="预期是否成功")


# ========== 工具函数 ==========
def safe_decode_image(base64_data: str) -> Optional[PILImage.Image]:
    """安全解码 base64 图像数据"""
    try:
        if not base64_data or not isinstance(base64_data, str):
            raise ValueError("无效的 base64 数据")

        decoded_data = base64.b64decode(base64_data)

        if len(decoded_data) == 0:
            raise ValueError("解码后数据为空")

        # 验证图像完整性
        image = PILImage.open(io.BytesIO(decoded_data))
        image.verify()

        # 重新打开图像
        image = PILImage.open(io.BytesIO(decoded_data))
        return image

    except Exception as e:
        print(f"❌ 图像解码失败: {e}")
        return None


def calculate_image_md5(base64_data: str) -> str:
    """计算base64图像数据的MD5值"""
    try:
        return hashlib.md5(base64_data.encode('utf-8')).hexdigest()
    except Exception as e:
        print(f"❌ MD5计算失败: {e}")
        return ""


def validate_image_path(image_path: str) -> bool:
    """验证图像路径是否存在且有效"""
    path_obj = Path(image_path)
    return path_obj.exists() and path_obj.is_file()


# Optional: create a sampling callback
async def handle_sampling_message(
        context: RequestContext[ClientSession, None], params: types.CreateMessageRequestParams
) -> types.CreateMessageResult:
    print(f"Sampling request: {params.messages}")
    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(
            type="text",
            text="Hello, world! from model",
        ),
        model="gpt-3.5-turbo",
        stopReason="endTurn",
    )


# ========== 核心功能类 ==========
class PerformanceTester:
    """性能测试器"""

    @staticmethod
    async def performance_test(
            session: ClientSession,
            call_func: Callable[[ClientSession, str, Dict[str, Any]], Awaitable[Any]],
            tool_name: str,
            tool_params: Dict[str, Any],
            iterations: int = ClientConfig.TEST_ITERATIONS,
            progress_interval: int = ClientConfig.PROGRESS_INTERVAL
    ) -> Tuple[List[float], PerformanceStats]:
        """
        执行工具性能测试
        """
        elapsed_times = []
        results = []

        print(f"🔄 开始进行{iterations}次{tool_name}调用测试...")

        for i in range(1, iterations + 1):
            print(f"✅ 第{i}次调用 {tool_name}")

            start_time = time.time()
            try:
                result = await call_func(session, tool_name, tool_params)
                end_time = time.time()

                elapsed_time = end_time - start_time
                elapsed_times.append(elapsed_time)
                results.append(result)

                print(f"💡 第{i}次调用结果: {result}")
                print(f'⏰ 第{i}次耗时: {elapsed_time:.3f}s\n')

            except Exception as e:
                print(f"❌ 第{i}次调用失败: {e}")
                end_time = time.time()
                elapsed_times.append(end_time - start_time)

            if i % progress_interval == 0:
                print(f'📈 已完成 {i}/{iterations} 次调用')

        stats = PerformanceTester._calculate_stats(elapsed_times)
        PerformanceTester._print_stats(tool_name, stats)

        return elapsed_times, stats

    @staticmethod
    def _calculate_stats(elapsed_times: List[float]) -> PerformanceStats:
        """计算统计信息"""
        if not elapsed_times:
            return PerformanceStats(
                average_time=0, max_time=0, min_time=0,
                total_time=0, call_count=0, std_dev=0, stability="无数据"
            )

        stats = {
            'average_time': statistics.mean(elapsed_times),
            'max_time': max(elapsed_times),
            'min_time': min(elapsed_times),
            'total_time': sum(elapsed_times),
            'call_count': len(elapsed_times)
        }

        # 计算标准差
        if len(elapsed_times) > 1:
            stats['std_dev'] = statistics.stdev(elapsed_times)
        else:
            stats['std_dev'] = 0

        # 评估稳定性
        stability_threshold = stats['average_time'] * 0.1
        if stats['std_dev'] < stability_threshold:
            stats['stability'] = "优秀"
        elif stats['std_dev'] < stats['average_time'] * 0.2:
            stats['stability'] = "良好"
        else:
            stats['stability'] = "一般"

        return PerformanceStats(**stats)

    @staticmethod
    def _print_stats(tool_name: str, stats: PerformanceStats):
        """打印统计信息"""
        print(f'\n📊 {tool_name} 详细性能统计:')
        print(f'   调用次数: {stats.call_count}')
        print(f'   平均耗时: {stats.average_time:.3f}s')
        print(f'   标准差: {stats.std_dev:.3f}s')
        print(f'   最大耗时: {stats.max_time:.3f}s')
        print(f'   最小耗时: {stats.min_time:.3f}s')
        print(f'   总耗时: {stats.total_time:.3f}s')
        print(f'   性能稳定性: {stats.stability}')


class TWIZoomInClient:
    """TWI缩放图像客户端"""

    def __init__(self, session: ClientSession):
        self.session = session
        self.performance_tester = PerformanceTester()

    async def test_zoom_in_image(self, params: ZoomInImageParams) -> Any:
        """
        测试缩放图像功能
        """
        # 验证图像路径
        if not validate_image_path(params.image_path):
            raise FileNotFoundError(f"图像文件不存在: {params.image_path}")

        # 读取并编码图像
        with open(params.image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        # 构建配置
        config = params.config or {
            "coords_max": 1000,
            "patch_size": 512,
            "min_edge": 5,
            "max_images": 12,
            "max_tokens": 8192,
            "zoom_in_img_str": " <image>",
            "tag": params.tag,
            "dbg_dump": params.dbg_dump
        }

        # 调用工具
        return await self.session.call_tool("zoom_in_image", {
            "format": params.image_format,
            "data": encoded_image,
            "shape": {
                "n": params.shape_n,
                "c": params.shape_c,
                "w": params.shape_w,
                "h": params.shape_h
            },
            "coord": {
                "x1": params.coord_x1,
                "y1": params.coord_y1,
                "x2": params.coord_x2,
                "y2": params.coord_y2
            },
            "config": config
        })

    async def test_detect_and_extract_coordinates(self, text: str) -> Dict[str, Any]:
        """
        测试检测和提取坐标功能
        """
        result_data = {}

        try:
            # 检测缩放区域
            detect_result = await self.session.call_tool("detect_zoom_in_region", {
                "text": text,
                "last_end": 0
            })

            logger.info(f"🔍 检测结果: {detect_result}")

            # 检查结果是否有效
            if (detect_result.structuredContent and
                    detect_result.structuredContent.get("contains_region") and
                    detect_result.structuredContent.get("error_code") == 0):

                # 提取坐标
                extract_result = await self.session.call_tool("extract_coordinates", {
                    "text": text,
                    "zoom_in_start": detect_result.structuredContent["zoom_in_start"],
                    "zoom_in_end": detect_result.structuredContent["zoom_in_end"]
                })

                result_data["extract_result"] = extract_result.structuredContent
                logger.info(f"📊 提取结果: {extract_result}")
            else:
                logger.warning("⚠️ 未检测到有效的缩放区域")

            result_data["detect_result"] = detect_result.structuredContent

        except Exception as e:
            logger.error(f"❌ 坐标检测测试失败: {e}")
            result_data["error"] = str(e)

        return result_data

    async def run_comprehensive_tests(self):
        """运行综合测试"""
        print("\n=== 开始综合测试 ===\n")

        # 测试坐标检测和提取
        await self._test_coordinate_detection()

        # 测试图像缩放
        await self._test_image_zooming()

    async def _test_coordinate_detection(self):
        """测试坐标检测"""
        print("\n🔍 测试坐标检测功能")

        test_cases = [
            TestCase(name="正常用例", text='hello, this is <|zoom_in_s|>[[120,55.5,400,160]]<|zoom_in_e|> a test case.',
                     expected_success=True),
            TestCase(name="缺失开始标记", text='hello, this is [120,55.5,400,160]]<|zoom_in_e|> a test case.',
                     expected_success=False),
            TestCase(name="格式错误", text='hello, this is <|zoom_in_s|>[120.55.5,400.160]]<|zoom_in_e|> a test case.',
                     expected_success=False),
        ]

        for test_case in test_cases:
            print(f"\n✨ 测试: {test_case.name}")
            print(f"📝 文本: {test_case.text}")

            try:
                result = await self.test_detect_and_extract_coordinates(test_case.text)
                print(f"✅ 结果: {result}")
            except Exception as e:
                print(f"❌ 测试失败: {e}")

    async def _test_image_zooming(self):
        """测试图像缩放"""
        print("\n🖼️ 测试图像缩放功能")

        test_images = [
            ZoomInImageParams(
                image_path="./test_data/358x441.jpg", shape_w=358, shape_h=441,
                coord_x1=10, coord_y1=10, coord_x2=348, coord_y2=431, tag="358x441.jpg"
            ),
            ZoomInImageParams(
                image_path="./test_data/960x540.jpg", shape_w=960, shape_h=540,
                coord_x1=10, coord_y1=10, coord_x2=950, coord_y2=530, tag="960x540.jpg"
            ),
            ZoomInImageParams(
                image_path="./test_data/1920x1080.jpg", shape_w=1920, shape_h=1080,
                coord_x1=10, coord_y1=10, coord_x2=1910, coord_y2=1070, tag="1920x1080.jpg"
            ),
        ]

        for params in test_images:
            print(f"\n🔧 测试图像: {params.tag}")

            try:
                result = await self.test_zoom_in_image(params)
                self._process_zoom_result(result, params.tag)
            except Exception as e:
                print(f"❌ 图像缩放测试失败: {e}")

    def _process_zoom_result(self, result: Any, tag: str):
        """处理缩放结果"""
        if result.isError:
            print(f"🔵 {tag} 缩放失败: {result}")
            return

        print(f"✅ {tag} 缩放成功")

        for content in result.content:
            if isinstance(content, types.ImageContent):
                print(f"📊 图像 {content.mimeType}: {len(content.data)} 字节")

                # 计算MD5
                md5_hash = calculate_image_md5(content.data)
                print(f"🔑 MD5: {md5_hash}")

                # 保存图像
                self._save_zoomed_image(content.data, tag)

    def _save_zoomed_image(self, base64_data: str, tag: str):
        """保存缩放后的图像"""
        try:
            image = safe_decode_image(base64_data)
            if image:
                output_path = f"./test_data/{tag}_zoomed_client.jpg"
                image.save(output_path)
                print(f"💾 图像已保存: {output_path}")
        except Exception as e:
            print(f"❌ 保存图像失败: {e}")


# ========== 异步调用函数 ==========
async def call_tool_func(session: ClientSession, tool_name: str, tool_params: Dict[str, Any]) -> Any:
    """调用工具的函数"""
    return await session.call_tool(tool_name, tool_params)


async def get_prompt_func(session: ClientSession, prompt_name: str, prompt_params: Dict[str, Any]) -> Any:
    """调用提示词的函数"""
    return await session.get_prompt(prompt_name, prompt_params)


async def read_resource_func(session: ClientSession, resource_url: str, params: Dict[str, Any]) -> Any:
    """调用资源的函数"""
    return await session.read_resource(AnyUrl(resource_url))


# ========== 主运行函数 ==========
async def run():
    """主运行函数"""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, sampling_callback=handle_sampling_message) as session:

            # 初始化连接
            await session.initialize()

            # 创建客户端实例
            client = TWIZoomInClient(session)

            try:
                # 列出可用功能
                await _list_available_features(session)

                # 运行综合测试
                await client.run_comprehensive_tests()

                # 性能测试
                # await _run_performance_tests(client)

            except Exception as e:
                print(f"❌ 测试过程中发生错误: {e}")


async def _list_available_features(session: ClientSession):
    """列出可用功能"""
    try:
        # 列出提示词
        prompts = await session.list_prompts()
        print(f"💡 可用提示词: {[p.name for p in prompts.prompts]}")
    except Exception as e:
        print(f'❌ 列出提示词失败: {e}')

    try:
        # 列出资源
        resources = await session.list_resources()
        print(f"💡 可用资源: {[r.uri for r in resources.resources]}")
    except Exception as e:
        print(f'❌ 列出资源失败: {e}')

    try:
        # 列出工具
        tools = await session.list_tools()
        print(f"💡 可用工具: {[t.name for t in tools.tools]}")
    except Exception as e:
        print(f'❌ 列出工具失败: {e}')


async def _run_performance_tests(client: TWIZoomInClient):
    """运行性能测试"""
    print("\n⚡ 开始性能测试")

    # 性能测试用例
    test_text = 'hello, this is <|zoom_in_s|>[[120,55.5,400,160]]<|zoom_in_e|> a test case.'

    # 检测区域性能测试
    await client.performance_tester.performance_test(
        session=client.session,
        call_func=call_tool_func,
        tool_name="detect_zoom_in_region",
        tool_params={"text": test_text, "last_end": 0},
        iterations=20  # 减少迭代次数以加快测试
    )

    # 提取坐标性能测试
    await client.performance_tester.performance_test(
        session=client.session,
        call_func=call_tool_func,
        tool_name="extract_coordinates",
        tool_params={"text": test_text, "zoom_in_start": 15, "zoom_in_end": 48},
        iterations=20
    )


def main():
    """客户端脚本入口点"""
    asyncio.run(run())


if __name__ == "__main__":
    main()
