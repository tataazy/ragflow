#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import base64
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import zipfile
import base64
from dataclasses import dataclass
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Optional, Dict, List, Tuple

import numpy as np
import pdfplumber
import requests
from PIL import Image
from strenum import StrEnum

from deepdoc.parser.pdf_parser import RAGFlowPdfParser

LOCK_KEY_pdfplumber = "global_shared_lock_pdfplumber"
if LOCK_KEY_pdfplumber not in sys.modules:
    sys.modules[LOCK_KEY_pdfplumber] = threading.Lock()

# todo 从配置中读取
image_pre_url = "http://172.24.72.178/v1/document/image/"

class MinerUContentType(StrEnum):
    IMAGE = "image"
    TABLE = "table"
    TEXT = "text"
    EQUATION = "equation"
    CODE = "code"
    LIST = "list"
    DISCARDED = "discarded"


# Mapping from language names to MinerU language codes
LANGUAGE_TO_MINERU_MAP = {
    'English': 'en',
    'Chinese': 'ch',
    'Traditional Chinese': 'chinese_cht',
    'Russian': 'east_slavic',
    'Ukrainian': 'east_slavic',
    'Indonesian': 'latin',
    'Spanish': 'latin',
    'Vietnamese': 'latin',
    'Japanese': 'japan',
    'Korean': 'korean',
    'Portuguese BR': 'latin',
    'German': 'latin',
    'French': 'latin',
    'Italian': 'latin',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Kannada': 'ka',
    'Thai': 'th',
    'Greek': 'el',
    'Hindi': 'devanagari',
    'Bulgarian': 'cyrillic',
}


class MinerUBackend(StrEnum):
    """MinerU processing backend options."""

    PIPELINE = "pipeline"  # Traditional multimodel pipeline (default)
    VLM_TRANSFORMERS = "vlm-transformers"  # Vision-language model using HuggingFace Transformers
    VLM_MLX_ENGINE = "vlm-mlx-engine"  # Faster, requires Apple Silicon and macOS 13.5+
    VLM_VLLM_ENGINE = "vlm-vllm-engine"  # Local vLLM engine, requires local GPU
    VLM_VLLM_ASYNC_ENGINE = "vlm-vllm-async-engine"  # Asynchronous vLLM engine, new in MinerU API
    VLM_LMDEPLOY_ENGINE = "vlm-lmdeploy-engine"  # LMDeploy engine
    VLM_HTTP_CLIENT = "vlm-http-client"  # HTTP client for remote vLLM server (CPU only)


class MinerULanguage(StrEnum):
    """MinerU supported languages for OCR (pipeline backend only)."""

    CH = "ch"  # Chinese
    CH_SERVER = "ch_server"  # Chinese (server)
    CH_LITE = "ch_lite"  # Chinese (lite)
    EN = "en"  # English
    KOREAN = "korean"  # Korean
    JAPAN = "japan"  # Japanese
    CHINESE_CHT = "chinese_cht"  # Chinese Traditional
    TA = "ta"  # Tamil
    TE = "te"  # Telugu
    KA = "ka"  # Kannada
    TH = "th"  # Thai
    EL = "el"  # Greek
    LATIN = "latin"  # Latin
    ARABIC = "arabic"  # Arabic
    EAST_SLAVIC = "east_slavic"  # East Slavic
    CYRILLIC = "cyrillic"  # Cyrillic
    DEVANAGARI = "devanagari"  # Devanagari


class MinerUParseMethod(StrEnum):
    """MinerU PDF parsing methods (pipeline backend only)."""

    AUTO = "auto"  # Automatically determine the method based on the file type
    TXT = "txt"  # Use text extraction method
    OCR = "ocr"  # Use OCR method for image-based PDFs


@dataclass
class ImageReference:
    """图片引用信息"""
    original_ref: str  # 原始引用路径
    alt_text: str      # 替代文本
    image_data: bytes  # 图片数据
    mime_type: str     # MIME类型
    start_pos: int     # 在文本中的起始位置
    end_pos: int       # 在文本中的结束位置


@dataclass
class ProcessedImage:
    """处理后的图片信息"""
    url: str           # 图片URL
    caption: str       # 图片说明
    ocr_text: str      # OCR识别文本
    original_url: str  # 原始URL
    start_pos: int     # 起始位置
    end_pos: int       # 结束位置


@dataclass
class MinerUParseOptions:
    """Options for MinerU PDF parsing."""

    backend: MinerUBackend = MinerUBackend.PIPELINE
    lang: Optional[MinerULanguage] = None  # language for OCR (pipeline backend only)
    method: MinerUParseMethod = MinerUParseMethod.AUTO
    server_url: Optional[str] = None
    delete_output: bool = True
    parse_method: str = "raw"
    formula_enable: bool = True
    table_enable: bool = True


class MinerUParser(RAGFlowPdfParser):
    def __init__(self, mineru_path: str = "mineru", mineru_api: str = "", mineru_server_url: str = ""):
        self.mineru_api = mineru_api.rstrip("/")
        self.mineru_server_url = mineru_server_url.rstrip("/")
        self.outlines = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.b64_data_uri_pattern = re.compile(r'^data:image/(\w+);base64,(.+)$')
        self.image_ref_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

    @staticmethod
    def _is_zipinfo_symlink(member: zipfile.ZipInfo) -> bool:
        return (member.external_attr >> 16) & 0o170000 == 0o120000

    def extract_image_references(self, markdown_content: str) -> List[ImageReference]:
        """从Markdown内容中提取图片引用"""
        references = []
        
        for match in self.image_ref_pattern.finditer(markdown_content):
            alt_text = match.group(1)
            image_path = match.group(2)
            start_pos = match.start()
            end_pos = match.end()
            
            references.append(ImageReference(
                original_ref=image_path,
                alt_text=alt_text,
                image_data=b'',  # 将在后续处理中填充
                mime_type='',    # 将在后续处理中填充
                start_pos=start_pos,
                end_pos=end_pos
            ))
        
        return references

    def decode_base64_image(self, b64_str: str) -> Tuple[bytes, str]:
        """解码base64图片数据"""
        try:
            if match := self.b64_data_uri_pattern.match(b64_str):
                # data URI格式: data:image/png;base64,...
                ext = match.group(1)
                b64_data = match.group(2)
                image_bytes = base64.b64decode(b64_data)
                mime_type = f"image/{ext}"
            else:
                # 纯base64数据
                image_bytes = base64.b64decode(b64_str)
                mime_type = "image/png"  # 默认PNG
                
            return image_bytes, mime_type
        except Exception as e:
            self.logger.warning(f"Failed to decode base64 image: {e}")
            return b'', ''

    def process_images_with_context(self, md_content: str, images_b64: Dict[str, str], 
                                  sections: List) -> Tuple[str, List[ProcessedImage]]:
        """处理图片引用，建立上下文关联"""
        # 提取图片引用
        image_refs = self.extract_image_references(md_content)
        
        # 处理base64图片数据
        processed_images = []
        updated_content = md_content
        
        for ref in image_refs:
            # 查找对应的base64数据
            image_key = ref.original_ref.replace('images/', '')
            if image_key in images_b64:
                image_data, mime_type = self.decode_base64_image(images_b64[image_key])
                if image_data:
                    # 创建处理后的图片信息
                    processed_img = ProcessedImage(
                        url=f"storage://{ref.original_ref}",  # 临时URL格式
                        caption=ref.alt_text,
                        ocr_text="",  # TODO: 添加OCR功能
                        original_url=ref.original_ref,
                        start_pos=ref.start_pos,
                        end_pos=ref.end_pos
                    )
                    
                    # 更新引用信息
                    ref.image_data = image_data
                    ref.mime_type = mime_type
                    processed_images.append(processed_img)
                    
                    self.logger.info(f"Processed image: {ref.original_ref}, size: {len(image_data)} bytes")
            
        return updated_content, processed_images

    def store_images(self, processed_images: List[ProcessedImage], image_data_map: Dict[str, bytes],
                    storage_path: str = None) -> List[ProcessedImage]:
        """存储图片并更新URL"""
        if not storage_path:
            storage_path = tempfile.mkdtemp(prefix="mineru_images_")
        
        stored_images = []
        
        for img in processed_images:
            if img.original_url in image_data_map:
                try:
                    # 生成文件名
                    filename = os.path.basename(img.original_url)
                    if not filename:
                        filename = f"image_{hash(img.original_url)}.png"
                    
                    # 保存图片
                    file_path = os.path.join(storage_path, filename)
                    with open(file_path, 'wb') as f:
                        f.write(image_data_map[img.original_url])
                    
                    # 更新URL为本地路径
                    img.url = file_path
                    stored_images.append(img)
                    
                    self.logger.info(f"Stored image: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to store image {img.original_url}: {e}")
            
        return stored_images

    def _extract_zip_no_root(self, zip_path, extract_to, root_dir):
        self.logger.info(f"[MinerU] Extract zip: zip_path={zip_path}, extract_to={extract_to}, root_hint={root_dir}")
        base_dir = Path(extract_to).resolve()
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            members = zip_ref.infolist()
            if not root_dir:
                if members and members[0].filename.endswith("/"):
                    root_dir = members[0].filename
                else:
                    root_dir = None
            if root_dir:
                root_dir = root_dir.replace("\\", "/")
                if not root_dir.endswith("/"):
                    root_dir += "/"

            for member in members:
                if member.flag_bits & 0x1:
                    raise RuntimeError(f"[MinerU] Encrypted zip entry not supported: {member.filename}")
                if self._is_zipinfo_symlink(member):
                    raise RuntimeError(f"[MinerU] Symlink zip entry not supported: {member.filename}")

                name = member.filename.replace("\\", "/")
                if root_dir and name == root_dir:
                    self.logger.info("[MinerU] Ignore root folder...")
                    continue
                if root_dir and name.startswith(root_dir):
                    name = name[len(root_dir) :]
                if not name:
                    continue
                if name.startswith("/") or name.startswith("//") or re.match(r"^[A-Za-z]:", name):
                    raise RuntimeError(f"[MinerU] Unsafe zip path (absolute): {member.filename}")

                parts = [p for p in name.split("/") if p not in ("", ".")]
                if any(p == ".." for p in parts):
                    raise RuntimeError(f"[MinerU] Unsafe zip path (traversal): {member.filename}")

                rel_path = os.path.join(*parts) if parts else ""
                dest_path = (Path(extract_to) / rel_path).resolve(strict=False)
                if dest_path != base_dir and base_dir not in dest_path.parents:
                    raise RuntimeError(f"[MinerU] Unsafe zip path (escape): {member.filename}")

                if member.is_dir():
                    os.makedirs(dest_path, exist_ok=True)
                    continue

                os.makedirs(dest_path.parent, exist_ok=True)
                with zip_ref.open(member) as src, open(dest_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    @staticmethod
    def _is_http_endpoint_valid(url, timeout=5):
        try:
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            return response.status_code in [200, 301, 302, 307, 308]
        except Exception:
            return False

    def check_installation(self, backend: str = "pipeline", server_url: Optional[str] = None) -> tuple[bool, str]:
        reason = ""

        valid_backends = ["pipeline", "vlm-http-client", "vlm-transformers", "vlm-vllm-engine", "vlm-mlx-engine", "vlm-vllm-async-engine", "vlm-lmdeploy-engine"]
        if backend not in valid_backends:
            reason = f"[MinerU] Invalid backend '{backend}'. Valid backends are: {valid_backends}"
            self.logger.warning(reason)
            return False, reason

        if not self.mineru_api:
            reason = "[MinerU] MINERU_APISERVER not configured."
            self.logger.warning(reason)
            return False, reason

        api_openapi = f"{self.mineru_api}/openapi.json"
        try:
            api_ok = self._is_http_endpoint_valid(api_openapi)
            self.logger.info(f"[MinerU] API openapi.json reachable={api_ok} url={api_openapi}")
            if not api_ok:
                reason = f"[MinerU] MinerU API not accessible: {api_openapi}"
                return False, reason
        except Exception as exc:
            reason = f"[MinerU] MinerU API check failed: {exc}"
            self.logger.warning(reason)
            return False, reason

        if backend == "vlm-http-client":
            resolved_server = server_url or self.mineru_server_url
            if not resolved_server:
                reason = "[MinerU] MINERU_SERVER_URL required for vlm-http-client backend."
                self.logger.warning(reason)
                return False, reason
            try:
                server_ok = self._is_http_endpoint_valid(resolved_server)
                self.logger.info(f"[MinerU] vlm-http-client server check reachable={server_ok} url={resolved_server}")
            except Exception as exc:
                self.logger.warning(f"[MinerU] vlm-http-client server probe failed: {resolved_server}: {exc}")

        return True, reason

    def _run_mineru(
        self, input_path: Path, output_dir: Path, options: MinerUParseOptions, callback: Optional[Callable] = None
    ) -> tuple[Path, list]:
        return self._run_mineru_api(input_path, output_dir, options, callback)

    def _run_mineru_api(
        self, input_path: Path, output_dir: Path, options: MinerUParseOptions, callback: Optional[Callable] = None
    ) -> tuple[Path, list]:
        pdf_file_path = str(input_path)

        if not os.path.exists(pdf_file_path):
            raise RuntimeError(f"[MinerU] PDF file not exists: {pdf_file_path}")

        pdf_file_name = Path(pdf_file_path).stem.strip()
        output_path = tempfile.mkdtemp(prefix=f"{pdf_file_name}_{options.method}_", dir=str(output_dir))
        output_zip_path = os.path.join(str(output_dir), f"{Path(output_path).name}.zip")

        data = {
            "output_dir": "./output",
            "lang_list": options.lang,
            "backend": options.backend,
            "parse_method": options.method,
            "formula_enable": options.formula_enable,
            "table_enable": options.table_enable,
            "server_url": None,
            "return_md": True,
            "return_middle_json": False,  # 与WeKnora保持一致，减少不必要数据传输
            "return_model_output": False,  # 与WeKnora保持一致
            "return_content_list": True,
            "return_images": True,
            "response_format_zip": False,  # 与WeKnora保持一致，返回JSON格式
            "start_page_id": 0,
            "end_page_id": 99999,
        }

        if options.server_url:
            data["server_url"] = options.server_url
        elif self.mineru_server_url:
            data["server_url"] = self.mineru_server_url

        self.logger.info(f"[MinerU] request {data=}")
        self.logger.info(f"[MinerU] request {options=}")

        headers = {"Accept": "application/json"}
        try:
            self.logger.info(f"[MinerU] invoke api: {self.mineru_api}/file_parse backend={options.backend} server_url={data.get('server_url')}")
            if callback:
                callback(0.20, f"[MinerU] invoke api: {self.mineru_api}/file_parse")
            with open(pdf_file_path, "rb") as pdf_file:
                files = {"files": (pdf_file_name + ".pdf", pdf_file, "application/pdf")}
                with requests.post(
                    url=f"{self.mineru_api}/file_parse",
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=1800,
                    stream=True,
                ) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "")
                    if content_type.startswith("application/zip"):
                        self.logger.info(f"[MinerU] zip file returned, saving to {output_zip_path}...")

                        if callback:
                            callback(0.30, f"[MinerU] zip file returned, saving to {output_zip_path}...")

                        with open(output_zip_path, "wb") as f:
                            response.raw.decode_content = True
                            shutil.copyfileobj(response.raw, f)

                        self.logger.info(f"[MinerU] Unzip to {output_path}...")
                        self._extract_zip_no_root(output_zip_path, output_path, pdf_file_name + "/")

                        if callback:
                            callback(0.40, f"[MinerU] Unzip to {output_path}...")
                        
                        # ZIP模式下仍然返回空outputs，让上层使用_read_output
                        return Path(output_path), []
                    else:
                        # 处理JSON响应 (与WeKnora保持一致)
                        self.logger.info(f"[MinerU] JSON response received, content-type: {content_type}")
                        
                        if callback:
                            callback(0.30, f"[MinerU] processing JSON response")
                        
                        # 读取JSON响应
                        response_data = response.json()
                        
                        # 解析MinerU JSON响应结构
                        # 实际结构: {"results": {"filename": {"md_content": "...", "images": {...}}}}
                        document_data = None
                        if 'results' in response_data and isinstance(response_data['results'], dict):
                            # 获取第一个（也是唯一的）文件结果
                            file_results = response_data['results']
                            if file_results:
                                first_file_key = list(file_results.keys())[0]
                                document_data = file_results[first_file_key]
                                self.logger.info(f"[MinerU] Using response path: results.{first_file_key}")
                        
                        if not document_data:
                            self.logger.warning("[MinerU] No valid document data found in response")
                            raise RuntimeError("[MinerU] Invalid response structure")
                        
                        # 提取markdown内容和图片
                        md_content = document_data.get('md_content', '')
                        images_data = document_data.get('images', {})
                        
                        # 添加诊断日志确认images_data的实际格式
                        if images_data:
                            self.logger.info(f"[MinerU] images_data类型: {type(images_data)}, 键数量: {len(images_data)}")
                            # 获取第一个样本查看实际数据格式
                            first_key = list(images_data.keys())[0] if images_data else None
                            first_value = images_data[first_key] if first_key else None
                            if first_value:
                                self.logger.info(f"[MinerU] 图片数据样本 - key: {first_key}, value类型: {type(first_value)}, value长度: {len(str(first_value)) if first_value else 0}")
                                self.logger.info(f"[MinerU] 图片数据样本 - value前100字符: {str(first_value)[:100]}")
                        
                        # 创建输出目录
                        output_path_obj = Path(output_path)
                        output_path_obj.mkdir(parents=True, exist_ok=True)
                        
                        # 保存markdown内容
                        md_file_path = output_path_obj / f"{pdf_file_name}.md"
                        with open(md_file_path, 'w', encoding='utf-8') as f:
                            f.write(md_content)
                        
                        # 处理图片数据
                        if images_data:
                            images_dir = output_path_obj / "images"
                            images_dir.mkdir(exist_ok=True)
                            
                            # 构建图片信息字典
                            images_info = {}
                            images_dir_name = "images"
                            
                            for img_path, img_data in images_data.items():
                                try:
                                    img_data_str = str(img_data) if img_data else ""
                                    self.logger.debug(f"[MinerU] 处理图片: {img_path}, 数据类型: {type(img_data)}, 长度: {len(img_data_str)}")
                                    
                                    # 处理不同格式的图片数据
                                    if isinstance(img_data, dict):
                                        # 如果是字典格式，检查是否包含file_path或base64字段
                                        self.logger.info(f"[MinerU] 图片数据为字典格式，包含字段: {list(img_data.keys())}")
                                        if 'base64' in img_data:
                                            img_data = img_data['base64']
                                        elif 'file_path' in img_data or 'path' in img_data:
                                            img_path = img_data.get('file_path') or img_data.get('path')
                                            self.logger.info(f"[MinerU] 图片为文件路径: {img_path}")
                                            # 文件路径已经在服务器上，不需要保存
                                            relative_path = f"{images_dir_name}/{Path(img_path).name}"
                                            images_info[relative_path] = img_data
                                            continue
                                        else:
                                            self.logger.warning(f"[MinerU] 未知字典格式: {img_data.keys()}")
                                            continue
                                    
                                    elif isinstance(img_data, str):
                                        # 字符串格式，可能是base64或文件路径
                                        if img_data.startswith('data:image/'):
                                            # data URI格式: data:image/png;base64,...
                                            base64_part = img_data.split(',', 1)[1]
                                        elif os.path.isabs(img_data) or ('/' in img_data and not img_data.startswith('http')):
                                            # 看起来像文件路径（绝对路径或相对路径，不是URL）
                                            self.logger.info(f"[MinerU] 检测到文件路径格式: {img_data}")
                                            # 文件路径已经在服务器上，不需要保存
                                            # 保持键名与MinerU原始返回一致（不带images/前缀）
                                            images_info[img_path] = img_data
                                            continue
                                        else:
                                            # 纯base64格式
                                            base64_part = img_data
                                    else:
                                        self.logger.warning(f"[MinerU] 未知图片数据类型: {type(img_data)}")
                                        continue
                                    
                                    # 解码并保存base64图片
                                    img_bytes = base64.b64decode(base64_part)
                                    img_save_path = images_dir / img_path
                                    img_save_path.parent.mkdir(parents=True, exist_ok=True)
                                    
                                    with open(img_save_path, 'wb') as img_file:
                                        img_file.write(img_bytes)
                                    
                                    # 存储图片信息到images_info，保持键名与MinerU原始返回一致（不带images/前缀）
                                    # 这样与md_content去掉images/后的路径保持一致
                                    images_info[img_path] = img_data
                                    self.logger.debug(f"[MinerU] 成功保存图片: {img_save_path}, 大小: {len(img_bytes)} bytes")
                                         
                                except Exception as e:
                                    self.logger.warning(f"[MinerU] Failed to process image {img_path}: {e}")
                                    continue
                        
                        # 保存图片信息到临时文件，供后续使用
                        if images_info:
                            images_info_path = output_path_obj / "_images_info.json"
                            with open(images_info_path, 'w', encoding='utf-8') as f:
                                json.dump(images_info, f, ensure_ascii=False)
                        
                        self.logger.info(f"[MinerU] JSON response processed successfully")
                        if callback:
                            callback(0.40, f"[MinerU] JSON response processed")
                        
                        # 直接返回处理好的数据，避免写文件再读取的冗余操作
                        virtual_outputs = [{
                            "type": "text",
                            "text": md_content,
                            "page_number": 1
                        }]
                        
                        # 如果有图片信息，直接使用第一次保存的文件路径创建IMAGE输出
                        if images_info:
                            self.logger.info(f"[MinerU] images_info包含 {len(images_info)} 个图片条目")
                            
                            for img_relative_path, img_data in images_info.items():
                                # 第一次遍历已经将图片保存为实际文件
                                # img_relative_path格式是 "images/xxx"，需要转换为实际文件路径
                                actual_img_path = output_path_obj / img_relative_path
                                
                                self.logger.debug(f"[MinerU] 检查图片路径: {actual_img_path}, exists={actual_img_path.exists()}")
                                
                                if actual_img_path.exists():
                                    virtual_outputs.append({
                                        "type": MinerUContentType.IMAGE,
                                        "image_caption": [f"图片: {img_relative_path}"],
                                        "image_footnote": [],
                                        "img_path": str(actual_img_path)  # 使用实际文件路径
                                    })
                                #else:
                                    #self.logger.warning(f"[MinerU] 图片文件不存在: {actual_img_path}")
                                    # 尝试列出output_path_obj的内容
                                    #if output_path_obj.exists():
                                        #files = list(output_path_obj.rglob('*'))
                                        #self.logger.warning(f"[MinerU] output_path内容: {[str(f.relative_to(output_path_obj)) for f in files if f.is_file()]}")

                        self.logger.info(f"[MinerU] 创建了 {len(virtual_outputs)} 个outputs (1 text + {len(virtual_outputs)-1} images)")
                        
                        return Path(output_path), virtual_outputs
                        
        except Exception as e:
            raise RuntimeError(f"[MinerU] api failed with exception {e}")


    def upload_and_replace_images_in_markdown(
        self,
        md_content,
        images_info: dict,
        tenant_id: Optional[str] = None,
        kb_id: Optional[str] = None,
        bucket: str = "imagetemps"
    ) -> Tuple[str, dict]:
        """
        上传markdown中的所有图片到MinIO，并替换引用为MinIO URL
    
        图片引用格式从：
            ![](images/xxx.jpg)
        替换为：
            ![](minio://bucket-objname)
    
        Args:
            md_content: markdown内容（支持str或可转换为str的对象）
            images_info: 图片信息字典 {relative_path: base64_data}
            tenant_id: 租户ID
            kb_id: 知识库ID
            bucket: MinIO bucket名称
    
        Returns:
            Tuple[str, dict]: (替换后的markdown, 图片映射表{original_path: minio_id})
        """
        import xxhash
        import logging
        from concurrent.futures import ThreadPoolExecutor
        
        logger = logging.getLogger(__name__)
        
        # 确保md_content是字符串
        if not isinstance(md_content, str):
            logger.warning(f"[MinerU] md_content类型错误: {type(md_content).__name__}, 尝试转换")
            md_content = str(md_content)
            
        logger.info(f"[MinerU] ========== upload_and_replace_images_in_markdown 开始 ==========")
        logger.info(f"[MinerU] tenant_id={tenant_id}, kb_id={kb_id}, bucket={bucket}")
        logger.info(f"[MinerU] images_info数量: {len(images_info)}")
        logger.info(f"[MinerU] markdown内容长度: {len(md_content)} 字符")

        # 图片引用正则
        img_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

        # 收集所有唯一图片引用
        found_refs = img_pattern.findall(md_content)
        logger.info(f"[MinerU] Markdown中发现图片引用总数: {len(found_refs)}")

        unique_refs = list(dict.fromkeys(found_refs))
        logger.info(f"[MinerU] 唯一图片引用数量: {len(unique_refs)}")

        # 详细打印每个图片引用
        #for idx, (alt_text, img_path) in enumerate(unique_refs):
            # images_info 的键不带 images/ 前缀，需要处理
            #img_key = img_path.replace('images/', '') if img_path.startswith('images/') else img_path
            #in_images_info = img_key in images_info
            #logger.info(f"[MinerU]   [{idx+1}] path='{img_path}' -> key='{img_key}' in_images_info={in_images_info}")

        # 准备上传任务
        upload_tasks = []
        image_mapping = {}  # img_path -> minio_id
        skipped_not_in_info = 0
        skipped_decode_error = 0

        for alt_text, img_path in unique_refs:
            # images_info 的键不带 images/ 前缀
            img_key = img_path.replace('images/', '') if img_path.startswith('images/') else img_path
            
            if img_key not in images_info:
                logger.warning(f"[MinerU] [跳过] 图片引用 '{img_path}' (key='{img_key}') 不在images_info中")
                skipped_not_in_info += 1
                continue

            if img_path in image_mapping:
                logger.debug(f"[MinerU] [跳过] 图片 '{img_path}' 已处理过")
                continue

            img_data = images_info[img_key]
            logger.debug(f"[MinerU] 处理图片 '{img_path}' (key='{img_key}'), 数据类型={type(img_data).__name__}, 长度={len(str(img_data)) if img_data else 0}")

            try:
                # 解码base64
                if img_data.startswith('data:image/'):
                    base64_part = img_data.split(',', 1)[1]
                    logger.debug(f"[MinerU]   提取base64部分，长度: {len(base64_part)}")
                else:
                    base64_part = img_data

                img_bytes = base64.b64decode(base64_part)
                logger.debug(f"[MinerU]   解码成功，图片字节数: {len(img_bytes)}")

                # 验证图片格式
                try:
                    img_obj = Image.open(BytesIO(img_bytes))
                    logger.debug(f"[MinerU]   图片格式: {img_obj.format}, 尺寸: {img_obj.size}")
                except Exception as img_err:
                    logger.warning(f"[MinerU]   图片格式验证失败: {img_err}")

                # 生成唯一的minio_id
                id_source = f"{tenant_id or ''}_{kb_id or ''}_{img_path}"
                logger.debug(f"[MinerU]   id_source: {id_source[:50]}...")

                obj_hash = xxhash.xxh64(id_source.encode()).hexdigest()
                objname = f"{tenant_id or 'ragflow'}_{obj_hash[:16]}"
                minio_id = f"{bucket}_{objname}"

                #logger.info(f"[MinerU]   -> minio_id: {minio_id}")

                image_mapping[img_path] = minio_id
                upload_tasks.append((img_path, img_bytes, minio_id, objname))

            except Exception as e:
                logger.warning(f"[MinerU] [跳过] 解码图片 '{img_path}' 失败: {e}")
                skipped_decode_error += 1
                continue

        logger.info(f"[MinerU] 上传任务准备完成: 总计={len(upload_tasks)}, 跳过(不在info)={skipped_not_in_info}, 跳过(解码失败)={skipped_decode_error}")

        # 批量上传
        upload_success = 0
        upload_failed = 0
        if upload_tasks:
            def upload_single(task):
                img_path, img_bytes, minio_id, objname = task
                nonlocal upload_success, upload_failed
                try:
                    from common import settings
                    #logger.info(f"[MinerU] [上传开始] {img_path} -> {objname}, 大小={len(img_bytes)} bytes")
                    settings.STORAGE_IMPL.put(bucket=bucket, fnm=objname, binary=img_bytes)
                    #logger.info(f"[MinerU] [上传成功] {objname}")
                    upload_success += 1
                    return True
                except Exception as e:
                    logger.warning(f"[MinerU] [上传失败] {objname}: {e}")
                    upload_failed += 1
                    return False

            try:
                logger.info(f"[MinerU] 开始批量上传，使用线程池(workers=8)")
                with ThreadPoolExecutor(max_workers=8) as executor:
                    results = list(executor.map(upload_single, upload_tasks))
                logger.info(f"[MinerU] 批量上传完成: 成功={upload_success}, 失败={upload_failed}")
            except Exception as e:
                logger.error(f"[MinerU] 批量上传异常: {e}")
                import traceback
                logger.error(f"[MinerU] 批量上传异常详情: {traceback.format_exc()}")
        else:
            logger.warning(f"[MinerU] 没有需要上传的图片任务")

        # 替换markdown引用
        logger.info(f"[MinerU] 开始替换markdown中的图片引用")
        original_img_count = len(img_pattern.findall(md_content))
        replaced_count = 0
        
        # 调试：检查 image_mapping 的内容
        logger.info(f"[MinerU] [DEBUG] image_mapping 大小: {len(image_mapping)}")
        if image_mapping:
            sample_keys = list(image_mapping.keys())[:3]
            logger.info(f"[MinerU] [DEBUG] image_mapping 键样本: {sample_keys}")
            logger.info(f"[MinerU] [DEBUG] 第一个键长度: {len(sample_keys[0]) if sample_keys else 0}")

        def replace_img_ref(match):
            nonlocal replaced_count
            # 获取原始匹配之前的上下文（用于检查换行）
            prefix = md_content[:match.start()]
            alt_text = match.group(1)
            img_path = match.group(2)

            if img_path in image_mapping:
                minio_id = image_mapping[img_path]
                new_ref = f"![{alt_text}]({image_pre_url}{minio_id})"
                replaced_count += 1
                logger.debug(f"[MinerU]   替换: {img_path} -> minio://{minio_id}")
                # 检查图片前是否有足够的换行（确保表格后图片能正确渲染）
                if prefix:
                    if prefix.endswith('>'):
                        # 可能是表格结束，添加换行确保渲染正确
                        return '\n\n' + new_ref
                    elif not prefix.endswith(('\n', '\r\n')):
                        # 如果前面不是换行结尾，添加换行
                        return '\n' + new_ref
                return new_ref
            logger.warning(f"[MinerU]   [未替换] 图片 '{img_path}' (长度={len(img_path)}) 不在映射表中 (映射表大小={len(image_mapping)})")
            return match.group(0)

        updated_md = img_pattern.sub(replace_img_ref, md_content)

        logger.info(f"[MinerU] 替换完成: 原文图片引用={original_img_count}, 成功替换={replaced_count}")
        logger.info(f"[MinerU] 更新后markdown长度: {len(updated_md)} 字符")

        # 打印替换后的markdown样本（用于调试）
        sample_updated = updated_md[:500] if updated_md else ""
        logger.info(f"[MinerU] 更新后markdown样本（前500字符）: {sample_updated}")

        logger.info(f"[MinerU] ========== upload_and_replace_images_in_markdown 结束 ==========")

        return updated_md, image_mapping



    def __images__(self, fnm, zoomin: int = 1, page_from=0, page_to=600, callback=None):
        self.page_from = page_from
        self.page_to = page_to
        try:
            with pdfplumber.open(fnm) if isinstance(fnm, (str, PathLike)) else pdfplumber.open(BytesIO(fnm)) as pdf:
                self.pdf = pdf
                self.page_images = [p.to_image(resolution=72 * zoomin, antialias=True).original for _, p in
                                    enumerate(self.pdf.pages[page_from:page_to])]
        except Exception as e:
            self.page_images = None
            self.total_page = 0
            self.logger.exception(e)

    def _line_tag(self, bx):
        # 容错处理：如果缺少page_idx，默认为第1页
        page_idx = bx.get("page_idx", 0)
        pn = [page_idx + 1]
        positions = bx.get("bbox", (0, 0, 0, 0))
        x0, top, x1, bott = positions
        # Normalize flipped coordinates (MinerU may report inverted bbox for flipped images)
        if x0 > x1:
            x0, x1 = x1, x0
        if top > bott:
            top, bott = bott, top

        if hasattr(self, "page_images") and self.page_images and len(self.page_images) > page_idx:
            page_width, page_height = self.page_images[page_idx].size
            x0 = (x0 / 1000.0) * page_width
            x1 = (x1 / 1000.0) * page_width
            top = (top / 1000.0) * page_height
            bott = (bott / 1000.0) * page_height

        return "@@{}\t{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}##".format("-".join([str(p) for p in pn]), x0, x1, top, bott)

    def crop(self, text, ZM=1, need_position=False):
        imgs = []
        poss = self.extract_positions(text)
        if not poss:
            if need_position:
                return None, None
            return

        if not getattr(self, "page_images", None):
            self.logger.warning("[MinerU] crop called without page images; skipping image generation.")
            if need_position:
                return None, None
            return

        page_count = len(self.page_images)

        filtered_poss = []
        for pns, left, right, top, bottom in poss:
            if not pns:
                self.logger.warning("[MinerU] Empty page index list in crop; skipping this position.")
                continue
            valid_pns = [p for p in pns if 0 <= p < page_count]
            if not valid_pns:
                self.logger.warning(f"[MinerU] All page indices {pns} out of range for {page_count} pages; skipping.")
                continue
            filtered_poss.append((valid_pns, left, right, top, bottom))

        poss = filtered_poss
        if not poss:
            self.logger.warning("[MinerU] No valid positions after filtering; skip cropping.")
            if need_position:
                return None, None
            return

        max_width = max(np.max([right - left for (_, left, right, _, _) in poss]), 6)
        GAP = 6
        pos = poss[0]
        first_page_idx = pos[0][0]
        poss.insert(0, ([first_page_idx], pos[1], pos[2], max(0, pos[3] - 120), max(pos[3] - GAP, 0)))
        pos = poss[-1]
        last_page_idx = pos[0][-1]
        if not (0 <= last_page_idx < page_count):
            self.logger.warning(
                f"[MinerU] Last page index {last_page_idx} out of range for {page_count} pages; skipping crop.")
            if need_position:
                return None, None
            return
        last_page_height = self.page_images[last_page_idx].size[1]
        poss.append(
            (
                [last_page_idx],
                pos[1],
                pos[2],
                min(last_page_height, pos[4] + GAP),
                min(last_page_height, pos[4] + 120),
            )
        )

        positions = []
        for ii, (pns, left, right, top, bottom) in enumerate(poss):
            right = left + max_width

            if bottom <= top:
                bottom = top + 2

            for pn in pns[1:]:
                if 0 <= pn - 1 < page_count:
                    bottom += self.page_images[pn - 1].size[1]
                else:
                    self.logger.warning(
                        f"[MinerU] Page index {pn}-1 out of range for {page_count} pages during crop; skipping height accumulation.")

            if not (0 <= pns[0] < page_count):
                self.logger.warning(
                    f"[MinerU] Base page index {pns[0]} out of range for {page_count} pages during crop; skipping this segment.")
                continue

            img0 = self.page_images[pns[0]]
            x0, y0, x1, y1 = int(left), int(top), int(right), int(min(bottom, img0.size[1]))
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:
                y0, y1 = y1, y0
            if x1 <= x0 or y1 <= y0:
                continue
            crop0 = img0.crop((x0, y0, x1, y1))
            imgs.append(crop0)
            if 0 < ii < len(poss) - 1:
                positions.append((pns[0] + self.page_from, x0, x1, y0, y1))

            bottom -= img0.size[1]
            for pn in pns[1:]:
                if not (0 <= pn < page_count):
                    self.logger.warning(
                        f"[MinerU] Page index {pn} out of range for {page_count} pages during crop; skipping this page.")
                    continue
                page = self.page_images[pn]
                x0, y0, x1, y1 = int(left), 0, int(right), int(min(bottom, page.size[1]))
                if x0 > x1:
                    x0, x1 = x1, x0
                if y0 > y1:
                    y0, y1 = y1, y0
                if x1 <= x0 or y1 <= y0:
                    bottom -= page.size[1]
                    continue
                cimgp = page.crop((x0, y0, x1, y1))
                imgs.append(cimgp)
                if 0 < ii < len(poss) - 1:
                    positions.append((pn + self.page_from, x0, x1, y0, y1))
                bottom -= page.size[1]

        if not imgs:
            if need_position:
                return None, None
            return

        height = 0
        for img in imgs:
            height += img.size[1] + GAP
        height = int(height)
        width = int(np.max([i.size[0] for i in imgs]))
        pic = Image.new("RGB", (width, height), (245, 245, 245))
        height = 0
        for ii, img in enumerate(imgs):
            if ii == 0 or ii + 1 == len(imgs):
                img = img.convert("RGBA")
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                overlay.putalpha(128)
                img = Image.alpha_composite(img, overlay).convert("RGB")
            pic.paste(img, (0, int(height)))
            height += img.size[1] + GAP

        if need_position:
            return pic, positions
        return pic

    @staticmethod
    def extract_positions(txt: str):
        poss = []
        for tag in re.findall(r"@@[0-9-]+\t[0-9.\t]+##", txt):
            pn, left, right, top, bottom = tag.strip("#").strip("@").split("\t")
            left, right, top, bottom = float(left), float(right), float(top), float(bottom)
            poss.append(([int(p) - 1 for p in pn.split("-")], left, right, top, bottom))
        return poss

    def _read_output(self, output_dir: Path, file_stem: str, method: str = "auto", backend: str = "pipeline") -> list[
        dict[str, Any]]:
        json_file = None
        subdir = None
        attempted = []

        # mirror MinerU's sanitize_filename to align ZIP naming
        def _sanitize_filename(name: str) -> str:
            sanitized = re.sub(r"[/\\\.]{2,}|[/\\]", "", name)
            sanitized = re.sub(r"[^\w.-]", "_", sanitized, flags=re.UNICODE)
            if sanitized.startswith("."):
                sanitized = "_" + sanitized[1:]
            return sanitized or "unnamed"

        safe_stem = _sanitize_filename(file_stem)
        allowed_names = {f"{file_stem}_content_list.json", f"{safe_stem}_content_list.json"}
        self.logger.info(f"[MinerU] Expected output files: {', '.join(sorted(allowed_names))}")
        self.logger.info(f"[MinerU] Searching output in: {output_dir}")

        jf = output_dir / f"{file_stem}_content_list.json"
        self.logger.info(f"[MinerU] Trying original path: {jf}")
        attempted.append(jf)
        if jf.exists():
            subdir = output_dir
            json_file = jf
        else:
            alt = output_dir / f"{safe_stem}_content_list.json"
            self.logger.info(f"[MinerU] Trying sanitized filename: {alt}")
            attempted.append(alt)
            if alt.exists():
                subdir = output_dir
                json_file = alt
            else:
                nested_alt = output_dir / safe_stem / f"{safe_stem}_content_list.json"
                self.logger.info(f"[MinerU] Trying sanitized nested path: {nested_alt}")
                attempted.append(nested_alt)
                if nested_alt.exists():
                    subdir = nested_alt.parent
                    json_file = nested_alt

        if not json_file:
            raise FileNotFoundError(f"[MinerU] Missing output file, tried: {', '.join(str(p) for p in attempted)}")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            for key in ("img_path", "table_img_path", "equation_img_path"):
                if key in item and item[key]:
                    item[key] = str((subdir / item[key]).resolve())
        
        # 读取额外的图片信息
        images_info_path = output_dir / "_images_info.json"
        if images_info_path.exists():
            try:
                with open(images_info_path, 'r', encoding='utf-8') as f:
                    images_info = json.load(f)
                # 将图片信息添加到返回数据中
                data.append({
                    "type": "_images_info",
                    "images": images_info
                })
                self.logger.info(f"[MinerU] Loaded {len(images_info)} images info")
            except Exception as e:
                self.logger.warning(f"[MinerU] Failed to load images info: {e}")
        
        return data

    def _transfer_to_sections(self, outputs: list[dict[str, Any]], parse_method: str = None):
        sections = []
        for output in outputs:
            match output["type"]:
                case MinerUContentType.TEXT:
                    section = output.get("text", "")
                case MinerUContentType.TABLE:
                    section = output.get("table_body", "") + "\n".join(output.get("table_caption", [])) + "\n".join(
                        output.get("table_footnote", []))
                    if not section.strip():
                        section = "FAILED TO PARSE TABLE"
                case MinerUContentType.IMAGE:
                    section = "".join(output.get("image_caption", [])) + "\n" + "".join(
                        output.get("image_footnote", []))
                case MinerUContentType.EQUATION:
                    section = output.get("text", "")
                case MinerUContentType.CODE:
                    section = output.get("code_body", "") + "\n".join(output.get("code_caption", []))
                case MinerUContentType.LIST:
                    section = "\n".join(output.get("list_items", []))
                case MinerUContentType.DISCARDED:
                    continue  # Skip discarded blocks entirely

            if section and parse_method == "manual":
                sections.append((section, output["type"], self._line_tag(output)))
            elif section and parse_method == "paper":
                sections.append((section + self._line_tag(output), output["type"]))
            else:
                sections.append((section, self._line_tag(output)))
        return sections

    def _transfer_to_tables(self, outputs: list[dict[str, Any]]):
        return []

    def parse_pdf(
            self,
            filepath: str | PathLike[str],
            binary: BytesIO | bytes,
            callback: Optional[Callable] = None,
            *,
            output_dir: Optional[str] = None,
            backend: str = "pipeline",
            server_url: Optional[str] = None,
            delete_output: bool = True,
            parse_method: str = "raw",
            return_images: bool = True,  # 新增参数
            return_section_images: bool = False,  # 新增参数
            **kwargs,
    ) -> tuple:
        import shutil

        temp_pdf = None
        created_tmp_dir = False

        parser_cfg = kwargs.get('parser_config', {})
        lang = parser_cfg.get('mineru_lang') or kwargs.get('lang', 'English')
        mineru_lang_code = LANGUAGE_TO_MINERU_MAP.get(lang, 'ch')  # Defaults to Chinese if not matched
        mineru_method_raw_str = parser_cfg.get('mineru_parse_method', 'auto')
        enable_formula = parser_cfg.get('mineru_formula_enable', True)
        enable_table = parser_cfg.get('mineru_table_enable', True)

        # remove spaces, or mineru crash, and _read_output fail too
        file_path = Path(filepath)
        pdf_file_name = file_path.stem.replace(" ", "") + ".pdf"
        pdf_file_path_valid = os.path.join(file_path.parent, pdf_file_name)

        if binary:
            temp_dir = Path(tempfile.mkdtemp(prefix="mineru_bin_pdf_"))
            temp_pdf = temp_dir / pdf_file_name
            with open(temp_pdf, "wb") as f:
                f.write(binary)
            pdf = temp_pdf
            self.logger.info(f"[MinerU] Received binary PDF -> {temp_pdf}")
            if callback:
                callback(0.15, f"[MinerU] Received binary PDF -> {temp_pdf}")
        else:
            if pdf_file_path_valid != filepath:
                self.logger.info(f"[MinerU] Remove all space in file name: {pdf_file_path_valid}")
                shutil.move(filepath, pdf_file_path_valid)
            pdf = Path(pdf_file_path_valid)
            if not pdf.exists():
                if callback:
                    callback(-1, f"[MinerU] PDF not found: {pdf}")
                raise FileNotFoundError(f"[MinerU] PDF not found: {pdf}")

        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = Path(tempfile.mkdtemp(prefix="mineru_pdf_"))
            created_tmp_dir = True

        self.logger.info(f"[MinerU] Output directory: {out_dir} backend={backend} api={self.mineru_api} server_url={server_url or self.mineru_server_url}")
        if callback:
            callback(0.15, f"[MinerU] Output directory: {out_dir}")

        self.__images__(pdf, zoomin=1)

        try:
            options = MinerUParseOptions(
                backend=MinerUBackend(backend),
                lang=MinerULanguage(mineru_lang_code),
                method=MinerUParseMethod(mineru_method_raw_str),
                server_url=server_url,
                delete_output=delete_output,
                parse_method=parse_method,
                formula_enable=enable_formula,
                table_enable=enable_table,
            )
            final_out_dir, outputs = self._run_mineru(pdf, out_dir, options, callback=callback)
            # 如果outputs为空或者需要从文件读取，则使用传统方式
            if not outputs:
                outputs = self._read_output(final_out_dir, pdf.stem, method=mineru_method_raw_str, backend=backend)
            self.logger.info(f"[MinerU] Parsed {len(outputs)} blocks from PDF.")
            if callback:
                callback(0.75, f"[MinerU] Parsed {len(outputs)} blocks from PDF.")

            # ========== 图片上传和引用替换（必须在生成sections之前） ==========
            # 直接在 TEXT output 中替换图片引用，不破坏原有结构
            self.logger.info(f"[MinerU] ========== 开始图片上传和引用替换 ==========")
            
            # 读取 images_info 文件
            images_info = {}
            images_info_path = final_out_dir / "_images_info.json"
            if images_info_path.exists():
                try:
                    with open(images_info_path, 'r', encoding='utf-8') as f:
                        images_info = json.load(f)
                    self.logger.info(f"[MinerU] 读取images_info: {len(images_info)} 个条目")
                except Exception as e:
                    self.logger.warning(f"[MinerU] 读取images_info失败: {e}")
            else:
                self.logger.warning(f"[MinerU] images_info文件不存在")
            
            # 获取 tenant_id 和 kb_id
            tenant_id = parser_cfg.get('tenant_id') or kwargs.get('tenant_id')
            kb_id = parser_cfg.get('kb_id') or kwargs.get('kb_id')
            self.logger.info(f"[MinerU] tenant_id={tenant_id}, kb_id={kb_id}")
            
            # 找到 TEXT output，获取 md_content
            md_content = None
            for output in outputs:
                if output.get("type") == MinerUContentType.TEXT:
                    md_content = output.get("text", "")
                    break
            
            if md_content and images_info:
                self.logger.info(f"[MinerU] 找到TEXT output，md_content长度={len(md_content)}, images_info数量={len(images_info)}")
                
                # 调用模块级函数上传图片并替换引用
                updated_md, image_mapping = self.upload_and_replace_images_in_markdown(
                    md_content=md_content,
                    images_info=images_info,
                    tenant_id=tenant_id,
                    kb_id=kb_id,
                    bucket="multi"
                )
                
                self.logger.info(f"[MinerU] upload_and_replace_images_in_markdown返回: image_mapping大小={len(image_mapping)}")
                
                # 更新 TEXT output
                if updated_md != md_content and image_mapping:
                    for output in outputs:
                        if output.get("type") == MinerUContentType.TEXT:
                            output["text"] = updated_md
                            self.logger.info(f"[MinerU] TEXT output已更新，替换了 {len(image_mapping)} 个图片引用")
                            break
                elif not image_mapping:
                    self.logger.warning(f"[MinerU] image_mapping为空，跳过替换")
            else:
                if not md_content:
                    self.logger.warning(f"[MinerU] 未找到TEXT output，跳过图片处理")
                if not images_info:
                    self.logger.warning(f"[MinerU] images_info为空，跳过图片处理")
            
            self.logger.info(f"[MinerU] ========== 图片上传和引用替换结束 ==========")
            
            # 处理sections（必须在图片替换之后，这样才能使用替换后的md_content）
            sections = self._transfer_to_sections(outputs, parse_method)
            
            # 如果需要返回图片信息
            if return_images:
                # 这里应该从MinerU的响应中提取图片数据
                # 目前简化处理，后续可以增强
                image_sections = []
                for output in outputs:
                    if output.get("type") == MinerUContentType.IMAGE:
                        # 提取图片相关信息
                        image_caption = "\n".join(output.get("image_caption", []))
                        image_footnote = "\n".join(output.get("image_footnote", []))
                        image_section = image_caption + image_footnote
                        if image_section.strip():
                            image_sections.append((image_section, "@IMAGE@"))
                
                # 合并文本和图片sections
                combined_sections = []
                text_idx = 0
                img_idx = 0
                
                # 简单的交替合并策略（可根据实际需求优化）
                while text_idx < len(sections) or img_idx < len(image_sections):
                    if text_idx < len(sections):
                        combined_sections.append(sections[text_idx])
                        text_idx += 1
                    if img_idx < len(image_sections):
                        combined_sections.append(image_sections[img_idx])
                        img_idx += 1
                
                sections = combined_sections

            # 如果需要返回section_images，则收集图片信息
            section_images = []
            if return_section_images:
                success_count = 0
                fail_reasons = {"empty_path": 0, "file_not_exist": 0, "open_failed": 0, "other_type": 0}
                
                for output in outputs:
                    if output.get("type") == MinerUContentType.IMAGE:
                        # 从保存的图片文件创建PIL Image对象
                        img_path = output.get("img_path", "")
                        
                        if not img_path:
                            fail_reasons["empty_path"] += 1
                            section_images.append(None)
                            continue
                            
                        if not os.path.exists(img_path):
                            fail_reasons["file_not_exist"] += 1
                            #self.logger.debug(f"[MinerU] 图片文件不存在: {img_path}")
                            section_images.append(None)
                            continue
                            
                        try:
                            img_obj = Image.open(img_path)
                            success_count += 1
                            section_images.append(img_obj)
                        except Exception as e:
                            fail_reasons["open_failed"] += 1
                            self.logger.warning(f"[MinerU] Failed to open image file {img_path}: {e}")
                            section_images.append(None)
                    else:
                        fail_reasons["other_type"] += 1
                        section_images.append(None)
                
                # 添加详细的诊断日志
                self.logger.info(f"[MinerU] 图片加载诊断: 总outputs={len(outputs)}, 成功={success_count}, 失败原因: 空路径={fail_reasons['empty_path']}, 文件不存在={fail_reasons['file_not_exist']}, 打开失败={fail_reasons['open_failed']}, 非图片类型={fail_reasons['other_type']}")
            
            # 调试信息
            tables_result = self._transfer_to_tables(outputs)
            self.logger.info(f"[MinerU] Returning: sections={len(sections)}, tables={len(tables_result)}, section_images={len(section_images) if section_images else 0}")
            
            # 始终返回4个值，与by_mineru的期望一致
            result = (sections, tables_result, section_images if return_section_images else None, self)
            self.logger.info(f"[MinerU] Return tuple length: {len(result)}")
            return result
        finally:
            if temp_pdf and temp_pdf.exists():
                try:
                    temp_pdf.unlink()
                    temp_pdf.parent.rmdir()
                except Exception:
                    pass
            if delete_output and created_tmp_dir and out_dir.exists():
                try:
                    shutil.rmtree(out_dir)
                except Exception:
                    pass


if __name__ == "__main__":
    parser = MinerUParser("mineru")
    ok, reason = parser.check_installation()
    print("MinerU available:", ok)

    filepath = ""
    with open(filepath, "rb") as file:
        outputs = parser.parse_pdf(filepath=filepath, binary=file.read())
        for output in outputs:
            print(output)
