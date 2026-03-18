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
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pdfplumber
import requests
from PIL import Image
from strenum import StrEnum

from deepdoc.parser.pdf_parser import RAGFlowPdfParser

LOCK_KEY_pdfplumber = "global_shared_lock_pdfplumber"
if LOCK_KEY_pdfplumber not in sys.modules:
    sys.modules[LOCK_KEY_pdfplumber] = threading.Lock()


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

    @staticmethod
    def _is_zipinfo_symlink(member: zipfile.ZipInfo) -> bool:
        return (member.external_attr >> 16) & 0o170000 == 0o120000

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



    def _run_mineru_api(
        self, input_path: Path, output_dir: Path, options: MinerUParseOptions, callback: Optional[Callable] = None
    ) -> tuple[str, dict[str, str]]:
        """Call MinerU API and return markdown content and images directly from JSON response.
        
        Returns:
            tuple: (md_content, images_dict) where images_dict is {filename: base64_data}
        """
        pdf_file_path = str(input_path)

        if not os.path.exists(pdf_file_path):
            raise RuntimeError(f"[MinerU] PDF file not exists: {pdf_file_path}")

        pdf_file_name = Path(pdf_file_path).stem.strip()

        data = {
            "output_dir": "./output",
            "lang_list": str(options.lang) if options.lang else "ch",
            "backend": str(options.backend),
            "parse_method": str(options.method),
            "formula_enable": str(options.formula_enable).lower(),
            "table_enable": str(options.table_enable).lower(),
            "server_url": None,
            "return_md": "true",
            "return_middle_json": "false",
            "return_model_output": "false",
            "return_content_list": "true",  # Also get content_list as fallback
            "return_images": "true",
            "response_format_zip": "false",  # Direct JSON response, no ZIP
            "start_page_id": "0",
            "end_page_id": "99999",
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
                resp = requests.post(
                    url=f"{self.mineru_api}/file_parse",
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=1800,
                )
                resp.raise_for_status()
                
            result = resp.json()
            self.logger.info("[MinerU] Api completed successfully.")
            
            # Debug: log response structure
            self.logger.info(f"[MinerU] Response type: {type(result)}")
            if isinstance(result, dict):
                self.logger.info(f"[MinerU] Response keys: {list(result.keys())}")
            else:
                self.logger.warning(f"[MinerU] Unexpected response type: {type(result)}, value: {str(result)[:500]}")
            if 'results' in result:
                self.logger.info(f"[MinerU] Results keys: {list(result['results'].keys())}")
                if 'document' in result['results']:
                    doc_keys = list(result['results']['document'].keys())
                    self.logger.info(f"[MinerU] Document keys: {doc_keys}")
                    if 'md_content' in result['results']['document']:
                        md_len = len(result['results']['document']['md_content'] or '')
                        self.logger.info(f"[MinerU] md_content length: {md_len}")
                    if 'images' in result['results']['document']:
                        img_count = len(result['results']['document']['images'] or {})
                        self.logger.info(f"[MinerU] images count: {img_count}")
                if 'files' in result['results']:
                    self.logger.info(f"[MinerU] Files keys: {list(result['results']['files'].keys())}")
            
            if callback:
                callback(0.50, "[MinerU] API response received, extracting content...")
            
            # Extract markdown and images from response
            # MinerU response schema: results.document.md_content and results.document.images
            md_content, images = self._extract_from_response(result)
            
            if callback:
                callback(0.60, f"[MinerU] Extracted markdown ({len(md_content)} chars) and {len(images)} images")
            
            return md_content, images
            
        except Exception as e:
            raise RuntimeError(f"[MinerU] api failed with exception {e}")

    def _extract_from_response(self, result: dict) -> tuple[str, dict[str, str]]:
        """Extract markdown content and images from MinerU API response.
        
        Supports multiple response formats:
        - Standard: results.document.* or results.files.*
        - Filename-based: results.{filename}.* (some MinerU versions)
        
        Returns:
            tuple: (md_content, images_dict) where images_dict is {filename: base64_data}
        """
        md_content = ""
        images = {}
        content_list = None
        
        if not isinstance(result, dict):
            self.logger.warning(f"[MinerU] Unexpected result type: {type(result)}")
            return md_content, images
        
        # Try to extract from results.document first, then results.files
        results = result.get("results", {})
        
        if not isinstance(results, dict):
            self.logger.warning(f"[MinerU] Unexpected results type: {type(results)}")
            return md_content, images
        
        # Check results.document (standard format)
        document = results.get("document", {})
        if document and isinstance(document, dict):
            md_content = document.get("md_content", "") or ""
            images = document.get("images", {}) or {}
            content_list = document.get("content_list")
            self.logger.info(f"[MinerU] Using response path: results.document")
        
        # Fallback to results.files (standard format)
        if not md_content and not images and not content_list:
            files = results.get("files", {})
            if files and isinstance(files, dict):
                md_content = files.get("md_content", "") or ""
                images = files.get("images", {}) or {}
                content_list = files.get("content_list")
                self.logger.info(f"[MinerU] Using response path: results.files")
        
        # Fallback: some MinerU versions return results.{filename}.*
        if not md_content and not images and not content_list:
            # Find first non-standard key that contains md_content or images
            for key, value in results.items():
                if key in ("document", "files"):
                    continue
                if isinstance(value, dict):
                    if "md_content" in value or "images" in value or "content_list" in value:
                        md_content = value.get("md_content", "") or ""
                        images = value.get("images", {}) or {}
                        content_list = value.get("content_list")
                        self.logger.info(f"[MinerU] Using response path: results.{key}")
                        break
        
        # If no md_content but have content_list, try to rebuild markdown
        if not md_content and content_list:
            # content_list might be a JSON string or a list
            if isinstance(content_list, str):
                try:
                    content_list = json.loads(content_list)
                    self.logger.info(f"[MinerU] Parsed content_list from JSON string, {len(content_list)} items")
                except json.JSONDecodeError as e:
                    self.logger.warning(f"[MinerU] Failed to parse content_list as JSON: {e}")
                    content_list = None
            
            if isinstance(content_list, list):
                self.logger.info(f"[MinerU] Rebuilding markdown from content_list ({len(content_list)} items)")
                md_content = self._content_list_to_markdown(content_list)
        
        # Handle different image formats
        processed_images = {}
        for img_name, img_data in images.items():
            # Image data might be base64 string or data URI
            if isinstance(img_data, str):
                processed_images[img_name] = img_data
            else:
                self.logger.warning(f"[MinerU] Unexpected image data type for {img_name}: {type(img_data)}")
        
        self.logger.info(f"[MinerU] Extracted {len(processed_images)} images from response")
        return md_content, processed_images
    
    def _content_list_to_markdown(self, content_list: list) -> str:
        """Convert MinerU content_list to markdown format.
        
        Args:
            content_list: List of content blocks from MinerU
            
        Returns:
            Markdown formatted string
        """
        md_parts = []
        
        for item in content_list:
            item_type = item.get("type", "")
            
            if item_type == "text":
                text = item.get("text", "")
                text_level = item.get("text_level", 1)
                if text:
                    # Add appropriate header level
                    if text_level <= 6:
                        md_parts.append(f"{'#' * text_level} {text}")
                    else:
                        md_parts.append(text)
                    md_parts.append("")
                    
            elif item_type == "table":
                table_body = item.get("table_body", "")
                if table_body:
                    md_parts.append(table_body)
                    md_parts.append("")
                    
            elif item_type == "image":
                img_path = item.get("img_path", "")
                caption = item.get("image_caption", "")
                if img_path:
                    caption_text = caption[0] if isinstance(caption, list) and caption else "image"
                    md_parts.append(f"![{caption_text}]({img_path})")
                    md_parts.append("")
                    
            elif item_type == "equation":
                text = item.get("text", "")
                if text:
                    md_parts.append(f"$${text}$$")
                    md_parts.append("")
                    
            elif item_type == "code":
                code_body = item.get("code_body", "")
                if code_body:
                    md_parts.append(f"```\n{code_body}\n```")
                    md_parts.append("")
                    
            elif item_type == "list":
                list_items = item.get("list_items", [])
                for li in list_items:
                    md_parts.append(f"- {li}")
                if list_items:
                    md_parts.append("")
        
        return "\n".join(md_parts)

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
        pn = [bx["page_idx"] + 1]
        positions = bx.get("bbox", (0, 0, 0, 0))
        x0, top, x1, bott = positions
        # Normalize flipped coordinates (MinerU may report inverted bbox for flipped images)
        if x0 > x1:
            x0, x1 = x1, x0
        if top > bott:
            top, bott = bott, top

        if hasattr(self, "page_images") and self.page_images and len(self.page_images) > bx["page_idx"]:
            page_width, page_height = self.page_images[bx["page_idx"]].size
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

    def _process_base64_images(self, md_content: str, images: dict[str, str], output_dir: Path) -> tuple[str, list[Image.Image]]:
        """Process base64 images from MinerU response and return PIL Image objects.
        
        Args:
            md_content: Markdown content with image references like ![](images/xxx.jpg)
            images: Dictionary of {filename: base64_data_or_datauri}
            output_dir: Directory to save processed images (for caching)
            
        Returns:
            tuple: (updated_md_content, pil_images_list)
        """
        import base64
        import uuid
        from pathlib import Path
        from io import BytesIO
        
        pil_images = []
        updated_md = md_content
        
        # Pattern to match data URI: data:image/{ext};base64,{data}
        b64_data_uri_pattern = re.compile(r'^data:image/(\w+);base64,(.+)$')
        
        for img_name, img_data in images.items():
            original_ref = f"images/{img_name}"
            
            # Skip if not referenced in markdown
            if original_ref not in md_content:
                continue
            
            # Parse base64 data
            img_bytes = None
            ext = "png"
            
            if m := b64_data_uri_pattern.match(img_data):
                ext = m.group(1)
                try:
                    img_bytes = base64.b64decode(m.group(2))
                except Exception as e:
                    self.logger.warning(f"[MinerU] Failed to decode base64 image {img_name}: {e}")
                    continue
            else:
                # Raw base64 without data URI prefix
                try:
                    img_bytes = base64.b64decode(img_data)
                except Exception as e:
                    self.logger.warning(f"[MinerU] Failed to decode raw base64 image {img_name}: {e}")
                    continue
                # Try to get extension from filename
                if '.' in img_name:
                    ext = img_name.rsplit('.', 1)[-1]
            
            if not img_bytes:
                continue
            
            try:
                # Load as PIL Image
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                pil_images.append(img)
                
                # Also save to temp directory for debugging
                new_name = f"{uuid.uuid4()}.{ext}"
                img_path = output_dir / new_name
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
                
                # Replace reference in markdown (use placeholder that won't be used)
                updated_md = updated_md.replace(original_ref, f"[Image:{len(pil_images)-1}]")
                
                self.logger.info(f"[MinerU] Processed image: {img_name} -> {img.size}")
                
            except Exception as e:
                self.logger.warning(f"[MinerU] Failed to process image {img_name}: {e}")
        
        return updated_md, pil_images



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
            **kwargs,
    ) -> tuple:
        """Parse PDF using MinerU API and return markdown content with images.
        
        Returns:
            tuple: (sections, tables, section_images) where 
                - sections is list of (text, tag) tuples
                - section_images is list of PIL Image objects (or None)
        """
        import shutil

        temp_pdf = None
        temp_img_dir = None

        parser_cfg = kwargs.get('parser_config', {})
        lang = parser_cfg.get('mineru_lang') or kwargs.get('lang', 'English')
        mineru_lang_code = LANGUAGE_TO_MINERU_MAP.get(lang, 'ch')  # Defaults to Chinese if not matched
        mineru_method_raw_str = parser_cfg.get('mineru_parse_method', 'auto')
        enable_formula = parser_cfg.get('mineru_formula_enable', True)
        enable_table = parser_cfg.get('mineru_table_enable', True)

        # remove spaces, or mineru crash
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

        # Create temp directory for images
        temp_img_dir = Path(tempfile.mkdtemp(prefix="mineru_img_"))
        
        self.logger.info(f"[MinerU] Parsing with backend={backend} api={self.mineru_api} server_url={server_url or self.mineru_server_url}")
        if callback:
            callback(0.15, "[MinerU] Starting PDF parsing...")

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
            
            # Call API and get markdown + images directly from JSON response
            md_content, images = self._run_mineru_api(pdf, temp_img_dir, options, callback=callback)
            
            if not md_content:
                self.logger.warning("[MinerU] No markdown content returned from API")
                self.section_images = []
                return [], []
            
            self.logger.info(f"[MinerU] Received markdown ({len(md_content)} chars) and {len(images)} images from API")
            if callback:
                callback(0.70, f"[MinerU] Received markdown ({len(md_content)} chars) and {len(images)} images")
            
            # Process base64 images and update markdown references
            updated_md, pil_images = self._process_base64_images(md_content, images, temp_img_dir)
            
            self.logger.info(f"[MinerU] Processed {len(pil_images)} images")
            if callback:
                callback(0.85, f"[MinerU] Processed {len(pil_images)} images, generating sections...")
            
            # Split markdown into sections for chunking
            # Use headers as natural section boundaries
            sections, section_images = self._split_markdown_to_sections(updated_md, pil_images)
            
            self.logger.info(f"[MinerU] Generated {len(sections)} sections from markdown")
            if callback:
                callback(0.95, f"[MinerU] Generated {len(sections)} sections")
            
            # Store section_images on self so caller can access it
            self.section_images = section_images
            # Return sections and tables (section_images accessed via self.section_images)
            return sections, []
            
        finally:
            # Cleanup temp files
            if temp_pdf and temp_pdf.exists():
                try:
                    temp_pdf.unlink()
                    temp_pdf.parent.rmdir()
                except Exception:
                    pass
            if delete_output and temp_img_dir and temp_img_dir.exists():
                try:
                    shutil.rmtree(temp_img_dir)
                except Exception:
                    pass

    def _split_markdown_to_sections(self, md_content: str, pil_images: list[Image.Image] = None) -> tuple:
        """Split markdown content into sections for chunking.
        
        Uses headers (# ## ###) as natural boundaries, but merges short sections
        to avoid fragmentation. Keeps images with their context.
        
        Args:
            md_content: Markdown content with [Image:X] placeholders
            pil_images: List of PIL Image objects extracted from the document
            
        Returns:
            tuple: (sections, section_images) where both are lists
        """
        pil_images = pil_images or []
        
        # Pattern to find image references [Image:X]
        img_ref_pattern = re.compile(r'\[Image:(\d+)\]')
        
        # Split by headers (lines starting with #)
        lines = md_content.split('\n')
        
        # First pass: collect raw sections split by headers, track images in each
        raw_sections = []
        current_section = []
        current_images = set()
        
        for line in lines:
            is_header = re.match(r'^#{1,6}\s', line)
            
            # Find image references in this line
            img_refs = img_ref_pattern.findall(line)
            for ref in img_refs:
                current_images.add(int(ref))
            
            if is_header and current_section:
                # Save current section
                section_text = '\n'.join(current_section).strip()
                if section_text:
                    raw_sections.append((section_text, current_images.copy()))
                current_section = [line]
                current_images = set()
            else:
                current_section.append(line)
        
        # Don't forget the last section
        if current_section:
            section_text = '\n'.join(current_section).strip()
            if section_text:
                raw_sections.append((section_text, current_images.copy()))
        
        # If no sections were created (no headers), return whole content
        if not raw_sections:
            if md_content.strip():
                # Return all images with the single section (if any)
                if pil_images:
                    combined = self._combine_images(pil_images)
                    return [(md_content.strip(), "")], [combined]
                return [(md_content.strip(), "")], [None]
            return [], []
        
        # Second pass: merge short sections (less than 300 chars) with next section
        MIN_SECTION_LENGTH = 300
        merged_sections = []
        merged_images = []
        pending_short = None
        pending_images = set()
        
        for section_text, section_imgs in raw_sections:
            if pending_short:
                # Merge pending short section with current
                section_text = pending_short + "\n\n" + section_text
                section_imgs = pending_images.union(section_imgs)
                pending_short = None
                pending_images = set()
            
            if len(section_text) < MIN_SECTION_LENGTH:
                # Hold short sections for merging
                pending_short = section_text
                pending_images = section_imgs
            else:
                merged_sections.append(section_text)
                merged_images.append(section_imgs)
        
        # Don't forget the last pending short section
        if pending_short:
            if merged_sections:
                # Merge with last section
                merged_sections[-1] = merged_sections[-1] + "\n\n" + pending_short
                merged_images[-1] = merged_images[-1].union(pending_images)
            else:
                merged_sections.append(pending_short)
                merged_images.append(pending_images)
        
        # Build section_images list - each section gets its own combined image
        section_images = []
        cleaned_sections = []
        
        for section_text, img_indices in zip(merged_sections, merged_images):
            # Remove [Image:X] placeholders from text
            cleaned_text = img_ref_pattern.sub('', section_text).strip()
            if not cleaned_text:
                continue
            cleaned_sections.append(cleaned_text)
            
            if img_indices and pil_images:
                # Get images for this section
                section_pil_images = []
                for idx in sorted(img_indices):
                    if 0 <= idx < len(pil_images):
                        section_pil_images.append(pil_images[idx])
                
                if section_pil_images:
                    # Combine images for this section only
                    combined = self._combine_images(section_pil_images)
                    section_images.append(combined)
                else:
                    section_images.append(None)
            else:
                section_images.append(None)
        
        # Ensure lists are same length
        min_len = min(len(cleaned_sections), len(section_images))
        cleaned_sections = cleaned_sections[:min_len]
        section_images = section_images[:min_len]
        
        # Convert to final format
        sections = [(text, "") for text in cleaned_sections if text.strip()]
        section_images = section_images[:len(sections)]
        
        img_count = len([i for i in section_images if i is not None])
        self.logger.info(f"[MinerU] Merged {len(raw_sections)} raw sections into {len(sections)} final sections with {img_count} image sections")
        return sections, section_images
    
    def _combine_images(self, images: list[Image.Image]) -> Image.Image:
        """Combine multiple images vertically into one."""
        if not images:
            return None
        if len(images) == 1:
            return images[0]
        
        # Calculate total height and max width
        total_height = sum(img.size[1] for img in images)
        max_width = max(img.size[0] for img in images)
        
        # Create new image
        combined = Image.new('RGB', (max_width, total_height), (255, 255, 255))
        
        # Paste images
        y_offset = 0
        for img in images:
            combined.paste(img, (0, y_offset))
            y_offset += img.size[1]
        
        return combined


if __name__ == "__main__":
    parser = MinerUParser("mineru")
    ok, reason = parser.check_installation()
    print("MinerU available:", ok)

    filepath = ""
    with open(filepath, "rb") as file:
        outputs = parser.parse_pdf(filepath=filepath, binary=file.read())
        for output in outputs:
            print(output)
