import base64
from enum import Enum
from typing import Any, Optional, Literal, Dict, List
from dataclasses import dataclass

import mcp.types as types
from mcp.types import CallToolResult
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
from transformers.image_transforms import resize
from PIL import Image as PILImage
import io
import logging

# 配置logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - S - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class ToolErrorCode(Enum):
    """工具错误码枚举"""
    SUCCESS = 0
    INVALID_INPUT = 1
    PARSE_ERROR = 2
    EXECUTION_ERROR = 3
    STOP_CONDITION_MET = 4
    UNKNOWN_ERROR = 99


class TWITools(str, Enum):
    """工具名称枚举"""
    DETECT_ZOOM_IN = "detect_zoom_in_region"
    EXTRACT_COORDINATES = "extract_coordinates"
    ZOOM_IN_IMG = "zoom_in_image"


class TWIPrompts(str, Enum):
    """提示词名称枚举"""
    TEST_ECHO_PROMPT = "echo_prompt"


# ========== 统一的工具参数模型 ==========
class DetectZoomInParam(BaseModel):
    """检测缩放区域参数"""
    text: str = Field(description="包含 '<|zoom_in_s|>' 和 '<|zoom_in_e|>' 标记的文本")
    last_end: int = Field(description="搜索起始位置")


class ExtractCoordinatesParam(BaseModel):
    """提取坐标参数"""
    text: str = Field(description="包含坐标信息的文本")
    zoom_in_start: int = Field(description="缩放起始位置")
    zoom_in_end: int = Field(description="缩放结束位置")


class ImageShapeParam(BaseModel):
    """图像形状参数"""
    n: int = Field(description="图像批次数量")
    c: int = Field(description="图像通道数")
    w: int = Field(description="图像宽度")
    h: int = Field(description="图像高度")


class ZoomInCoordParam(BaseModel):
    """缩放坐标参数"""
    x1: float = Field(description="左上角x坐标")
    y1: float = Field(description="左上角y坐标")
    x2: float = Field(description="右下角x坐标")
    y2: float = Field(description="右下角y坐标")


class ZoomInImageParam(BaseModel):
    """缩放图像参数"""
    format: str = Field(description="图像格式，如 nv12, jpg, png")
    data: str = Field(description="base64编码的图像数据")
    shape: ImageShapeParam = Field(description="图像形状信息")
    coord: ZoomInCoordParam = Field(description="缩放坐标")
    config: Optional[Dict[str, Any]] = Field(default=None, description="缩放配置参数")


# ========== 统一的工具结果模型 ==========
class BaseToolResult(BaseModel):
    """基础工具结果模型"""
    error_code: int = Field(default=ToolErrorCode.SUCCESS.value, description="错误码")
    message: str = Field(default="success", description="结果消息")


class DetectZoomInResult(BaseToolResult):
    """检测缩放区域结果"""
    zoom_in_start: int = Field(description="缩放起始位置")
    zoom_in_end: int = Field(description="缩放结束位置")
    contains_region: bool = Field(description="是否包含缩放区域")


class ExtractCoordinatesResult(BaseToolResult):
    """提取坐标结果"""
    x1: Optional[float] = Field(default=None, description="左上角x坐标")
    y1: Optional[float] = Field(default=None, description="左上角y坐标")
    x2: Optional[float] = Field(default=None, description="右下角x坐标")
    y2: Optional[float] = Field(default=None, description="右下角y坐标")


@dataclass
class ZoomInCoordinate:
    """缩放坐标数据类"""
    x1: float
    y1: float
    x2: float
    y2: float


class TWIZoomInConfig(BaseModel):
    """缩放配置参数"""
    coords_max: int = Field(default=1000, ge=1, description="坐标最大值")
    patch_size: int = Field(default=512, ge=1, description="目标补丁大小")
    min_edge: int = Field(default=5, ge=1, description="最小边缘长度")
    max_images: int = Field(default=12, ge=1, description="最大图像数量")
    max_tokens: int = Field(default=8192, ge=1, description="最大token数量")
    zoom_in_img_str: str = Field(default=" <image>", description="缩放图像标记")
    tag: str = Field(default="NotSetTag", description="调试标签")
    dbg_dump: bool = Field(default=False, description="是否保存调试图像")


class TWIImgInfo(BaseModel):
    """图像信息"""
    width: int = Field(description="图像宽度")
    height: int = Field(description="图像高度")
    channels: int = Field(description="图像通道数")
    mode: Literal['L', 'RGB', 'RGBA', 'CMYK', 'YCbCr', 'LAB', 'HSV', 'I', 'F'] = Field(description="图像模式")
    dtype: str = Field(default='uint8', description="数据类型")
    format: Optional[str] = Field(default=None, description="图像格式")
    size_bytes: Optional[int] = Field(default=None, description="图像大小(字节)")

    @classmethod
    def from_array(cls, array: np.ndarray, mode: Optional[str] = None) -> "TWIImgInfo":
        """从numpy数组创建图像信息"""
        if len(array.shape) == 2:
            height, width = array.shape
            channels = 1
        else:
            height, width, channels = array.shape

        if mode is None:
            if channels == 1:
                mode = 'L'
            elif channels == 3:
                mode = 'RGB'
            elif channels == 4:
                mode = 'RGBA'
            else:
                mode = f'UNKNOWN_{channels}'

        return cls(
            width=width,
            height=height,
            channels=channels,
            mode=mode,
            dtype=str(array.dtype)
        )

    @classmethod
    def from_pil_image(cls, image: PILImage.Image) -> "TWIImgInfo":
        """从PIL图像创建图像信息"""
        return cls(
            width=image.width,
            height=image.height,
            channels=len(image.getbands()),
            mode=image.mode,
            dtype='uint8'
        )

    def validate_for_reconstruction(self) -> bool:
        """验证信息是否足以重建图像"""
        return all(hasattr(self, field) for field in ['width', 'height', 'mode'])


class TWIImgResult(BaseToolResult):
    """图像处理结果"""
    image_infos: List[TWIImgInfo] = Field(default_factory=list, description="图像信息列表")
    # image_contents: List[types.ImageContent] = Field(default_factory=list, description="图像内容列表")
    image_contents: List[Any] = Field(default_factory=list, description="图像内容列表")


class MCPResultConverter:
    """MCP 结果转换工具类"""

    @staticmethod
    def model_to_tool_result(model: BaseModel, is_error: bool = False) -> types.CallToolResult:
        """将Pydantic模型转换为符合MCP协议的ToolResult"""
        try:
            if hasattr(model, 'model_dump'):
                data = model.model_dump()

                # 过滤掉None值，避免JSON序列化问题
                filtered_data = {k: v for k, v in data.items() if v is not None}

                logger.info(f'===> data: {data} filtered_data: {filtered_data}')

                # 创建文本内容 - 使用正确的MCP类型
                text_content = "\n".join([f"{key}: {value}" for key, value in filtered_data.items()])

                # 返回符合MCP协议的结果
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=text_content)],
                    structuredContent=filtered_data,  # 关键：必须设置structuredContent
                    isError=is_error
                )

            else:
                # 如果模型没有dump方法，返回简单文本
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=str(model))],
                    structuredContent={"raw_data": str(model)},
                    isError=is_error
                )

        except Exception as e:
            logger.error(f"结果转换异常: {e}")
            error_data = {
                "error_code": ToolErrorCode.UNKNOWN_ERROR.value,
                "message": f"结果转换异常: {str(e)}"
            }
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"结果转换错误: {str(e)}")],
                structuredContent=error_data,
                isError=True
            )

    @staticmethod
    def error_result(message: str, error_code: int = ToolErrorCode.UNKNOWN_ERROR.value) -> types.CallToolResult:
        """创建符合MCP协议的错误结果"""
        try:
            error_data = {
                "error_code": error_code,
                "message": message
            }
            return MCPResultConverter.model_to_tool_result(
                BaseToolResult(**error_data),
                is_error=True
            )
        except Exception as e:
            logger.error(f"错误结果创建异常: {e}")
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=message)],
                structuredContent={"error": message},
                isError=True
            )

    @staticmethod
    def success_result(message: str = "Success") -> types.CallToolResult:
        """创建成功结果"""
        success_data = BaseToolResult(message=message)
        return MCPResultConverter.model_to_tool_result(
            success_data,
            is_error=False
        )


class MCPTWI:
    """TWI MCP 工具处理类"""

    def __init__(self):
        pass

    def detect_zoom_in_region(self, text: str, last_end: int) -> DetectZoomInResult:
        """检测缩放区域"""
        invalid_index = -1
        try:
            logger.info(f"🔍 开始检测缩放区域, text: {text}, last_end: {last_end}")

            zoom_in_start = text.find('<|zoom_in_s|>', last_end)
            zoom_in_end = text.find('<|zoom_in_e|>', zoom_in_start)

            logger.info(f"📌 找到标记位置: start={zoom_in_start}, end={zoom_in_end}")

            if zoom_in_start < 0 or zoom_in_end < zoom_in_start:
                logger.info("❌ 未找到有效的缩放区域标记")
                return DetectZoomInResult(
                    zoom_in_start=invalid_index,
                    zoom_in_end=invalid_index,
                    contains_region=False,
                    message='未检测到缩放区域'
                )

            logger.info("✅ 成功检测到缩放区域")
            return DetectZoomInResult(
                zoom_in_start=zoom_in_start,
                zoom_in_end=zoom_in_end,
                contains_region=True,
                message='检测到缩放区域'
            )

        except Exception as e:
            logger.error(f"❌ 检测缩放区域异常: {e}")
            return DetectZoomInResult(
                zoom_in_start=invalid_index,
                zoom_in_end=invalid_index,
                contains_region=False,
                error_code=ToolErrorCode.UNKNOWN_ERROR.value,
                message=f'检测缩放区域失败: {str(e)}'
            )

    def extract_coordinates(self, text: str, zoom_in_start: int, zoom_in_end: int) -> ExtractCoordinatesResult:
        """从文本中提取坐标"""
        start = text.find('[[', zoom_in_start)
        end = text.find(']]', start)

        if zoom_in_start <= start < end < zoom_in_end:
            try:
                coords = text[start + 2:end].split(',')
                x1, y1, x2, y2 = [float(x.strip()) for x in coords]
                return ExtractCoordinatesResult(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    message='坐标提取成功'
                )
            except Exception as e:
                logger.error(f'坐标提取失败: {e}')
                return ExtractCoordinatesResult(
                    error_code=ToolErrorCode.UNKNOWN_ERROR.value,
                    message=f'坐标提取失败: {str(e)}'
                )

        return ExtractCoordinatesResult(
            message=f'未找到有效坐标. Text is {text}'
        )

    def zoom_in_image_jpg(self, image: PILImage.Image, coord: ZoomInCoordinate,
                          config: Optional[TWIZoomInConfig] = None) -> TWIImgResult:
        """JPEG图像缩放处理"""
        try:
            if config is None:
                config = TWIZoomInConfig()

            # 调试图像保存
            if config.dbg_dump:
                image.save(f'./test_data/{config.tag}_server_img_ori.jpg')

            orig_image = np.array(image)
            max_edge = max(orig_image.shape[:2])
            logger.info(f'最大边缘: {max_edge}, 原始图像形状: {orig_image.shape[:2]}')

            # 坐标转换和验证
            coords = [coord.x1, coord.y1, coord.x2, coord.y2]
            x1, y1, x2, y2 = [int(int(x) / config.coords_max * max_edge) for x in coords]
            logger.info(f'原始坐标: {coords}, 转换后坐标: {x1, y1, x2, y2}')

            # 提取图像补丁
            patch = orig_image[y1:y2, x1:x2]
            patch_h, patch_w = patch.shape[:2]
            logger.info(f'补丁尺寸: {patch_h}x{patch_w}')

            # 缩放处理
            scale_factor = config.patch_size / max(patch_h, patch_w)
            logger.info(f'缩放因子: {scale_factor}')

            if scale_factor < 1:
                new_height = int(patch_h * scale_factor)
                new_width = int(patch_w * scale_factor)
                patch = resize(patch, size=(new_height, new_width))
                logger.info(f'缩放后尺寸: {patch.shape}')

            # 调试保存
            if config.dbg_dump:
                pil_image = PILImage.fromarray(patch.astype(np.uint8))
                pil_image.save(f'./test_data/{config.tag}_server_img_patch.jpg')

            # Base64编码
            processed_image = patch.astype(np.uint8).tobytes()
            encoded_data = base64.b64encode(processed_image).decode('utf-8')
            patch_img_info = TWIImgInfo.from_array(patch)

            # 验证编码解码
            if config.dbg_dump and patch_img_info.validate_for_reconstruction():
                decoded_data = base64.b64decode(encoded_data)
                decoded_image = PILImage.frombytes(
                    patch_img_info.mode,
                    (patch_img_info.width, patch_img_info.height),
                    decoded_data
                )
                decoded_image.save(f'./test_data/{config.tag}_server_img_patch_enc_dec.jpg')

            return TWIImgResult(
                image_infos=[patch_img_info],
                image_contents=[types.ImageContent(type="image", data=encoded_data, mimeType="image/jpeg")]
            )

        except Exception as e:
            logger.error(f'图像缩放处理异常: {e}')
            return TWIImgResult(
                image_infos=[TWIImgInfo(width=0, height=0, channels=0, mode='L')],
                image_contents=[types.ImageContent(type="image", data="", mimeType="image/jpeg")],
                error_code=ToolErrorCode.EXECUTION_ERROR.value,
                message=f"图像处理异常: {e}"
            )

    @staticmethod
    def _validate_coordinates(coord: ZoomInCoordinate, coords_max: int) -> bool:
        """验证坐标有效性"""
        return (coord.x1 < coord.x2 and
                coord.y1 < coord.y2 and
                all(0 <= getattr(coord, attr) <= coords_max
                    for attr in ['x1', 'y1', 'x2', 'y2']))

    @staticmethod
    def _clamp_coordinates(x1: int, y1: int, x2: int, y2: int, image_shape: tuple) -> tuple:
        """确保坐标在图像范围内"""
        height, width = image_shape[:2]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        return x1, y1, x2, y2


# ========== 工具定义和处理器 ==========
class ToolDefinition:
    """工具定义类"""

    @staticmethod
    def get_tool_definitions() -> List[types.Tool]:
        """获取所有工具定义"""
        return [
            types.Tool(
                name=TWITools.DETECT_ZOOM_IN,
                description="检测文本中的缩放区域标记",
                inputSchema=DetectZoomInParam.model_json_schema(),
                outputSchema=DetectZoomInResult.model_json_schema()
            ),
            types.Tool(
                name=TWITools.EXTRACT_COORDINATES,
                description="从文本中提取坐标信息",
                inputSchema=ExtractCoordinatesParam.model_json_schema(),
                outputSchema=ExtractCoordinatesResult.model_json_schema()
            ),
            types.Tool(
                name=TWITools.ZOOM_IN_IMG,
                description="根据坐标缩放图像",
                inputSchema=ZoomInImageParam.model_json_schema(),
                outputSchema=TWIImgResult.model_json_schema()
            ),
        ]


class ToolHandler:
    """工具处理器"""

    def __init__(self):
        self.mcp_twi = MCPTWI()
        self.converter = MCPResultConverter()

    async def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> dict[str, Any] | types.CallToolResult:
        """处理工具调用"""
        try:
            logger.info(f"🛠️ 处理工具调用: {name}, 参数: {arguments}")

            match name:
                case TWITools.DETECT_ZOOM_IN:
                    logger.info("🔍 开始处理 detect_zoom_in_region")
                    param = DetectZoomInParam(**arguments)
                    result = self.mcp_twi.detect_zoom_in_region(param.text, param.last_end)
                    logger.info(f"✅ detect_zoom_in_region 结果: {result}")
                    return result.model_dump()

                case TWITools.EXTRACT_COORDINATES:
                    logger.info("🔍 开始处理 extract_coordinates")
                    param = ExtractCoordinatesParam(**arguments)
                    result = self.mcp_twi.extract_coordinates(param.text, param.zoom_in_start, param.zoom_in_end)
                    logger.info(f"✅ extract_coordinates 结果: {result}")
                    return result.model_dump()

                case TWITools.ZOOM_IN_IMG:
                    logger.info("🖼️ 开始处理 zoom_in_image")
                    param = ZoomInImageParam(**arguments)

                    # 处理配置参数
                    config_value = param.config or {}
                    zoom_in_config = TWIZoomInConfig(**config_value)
                    zoom_in_coord = ZoomInCoordinate(**param.coord.model_dump())

                    logger.info(f'缩放参数: 格式={param.format}, 标签={zoom_in_config.tag}')

                    # 格式处理
                    if param.format == "jpg":
                        decoded_data = base64.b64decode(param.data)
                        decoded_image = PILImage.open(io.BytesIO(decoded_data))
                        result = self.mcp_twi.zoom_in_image_jpg(decoded_image, zoom_in_coord, zoom_in_config)
                        logger.info(f"✅ zoom_in_image 结果: 成功处理图像")

                        return self.converter.model_to_tool_result(result)
                    # elif format == "nv12":
                    #     # result = mcp_twi.zoom_in_image_nv12(...)
                    #     pass
                    else:
                        error_msg = f"不支持的图像格式: {param.format}"
                        logger.error(error_msg)
                        return self.converter.error_result(
                            error_msg,
                            ToolErrorCode.INVALID_INPUT.value
                        )

                case _:
                    error_msg = f"未知工具: {name}"
                    logger.error(error_msg)
                    return self.converter.error_result(
                        error_msg,
                        ToolErrorCode.INVALID_INPUT.value
                    )

        except Exception as e:
            error_msg = f"工具处理异常: {str(e)}"
            logger.error(error_msg)
            return self.converter.error_result(error_msg)
