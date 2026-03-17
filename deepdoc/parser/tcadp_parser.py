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
import tempfile
import time
import traceback
import types
import zipfile
from datetime import datetime
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Optional

import requests
from PIL import Image
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.lkeap.v20240522 import lkeap_client, models

from common.config_utils import get_base_config
from deepdoc.parser.pdf_parser import RAGFlowPdfParser


class TencentCloudAPIClient:
    """Tencent Cloud API client using official SDK"""

    def __init__(self, secret_id, secret_key, region):
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.region = region
        self.outlines = []

        # Create credentials
        self.cred = credential.Credential(secret_id, secret_key)

        # Instantiate an http option, optional, can be skipped if no special requirements
        self.httpProfile = HttpProfile()
        self.httpProfile.endpoint = "lkeap.tencentcloudapi.com"

        # Instantiate a client option, optional, can be skipped if no special requirements
        self.clientProfile = ClientProfile()
        self.clientProfile.httpProfile = self.httpProfile

        # Instantiate the client object for the product to be requested, clientProfile is optional
        self.client = lkeap_client.LkeapClient(self.cred, region, self.clientProfile)

    def reconstruct_document_sse(self, file_type, file_url=None, file_base64=None, file_start_page=1, file_end_page=1000, config=None):
        """Call document parsing API using official SDK"""
        try:
            # Instantiate a request object, each interface corresponds to a request object
            req = models.ReconstructDocumentSSERequest()

            # Build request parameters
            params = {
                "FileType": file_type,
                "FileStartPageNumber": file_start_page,
                "FileEndPageNumber": file_end_page,
            }

            # According to Tencent Cloud API documentation, either FileUrl or FileBase64 parameter must be provided, if both are provided only FileUrl will be used
            if file_url:
                params["FileUrl"] = file_url
                logging.info(f"[TCADP] Using file URL: {file_url}")
            elif file_base64:
                params["FileBase64"] = file_base64
                logging.info(f"[TCADP] Using Base64 data, length: {len(file_base64)} characters")
            else:
                raise ValueError("Must provide either FileUrl or FileBase64 parameter")

            if config:
                params["Config"] = config

            req.from_json_string(json.dumps(params))

            # The returned resp is an instance of ReconstructDocumentSSEResponse, corresponding to the request object
            resp = self.client.ReconstructDocumentSSE(req)
            parser_result = {}

            # Output json format string response
            if isinstance(resp, types.GeneratorType):  # Streaming response
                logging.info("[TCADP] Detected streaming response")
                for event in resp:
                    logging.info(f"[TCADP] Received event: {event}")
                    if event.get('data'):
                        try:
                            data_dict = json.loads(event['data'])
                            logging.info(f"[TCADP] Parsed data: {data_dict}")

                            if data_dict.get('Progress') == "100":
                                parser_result = data_dict
                                logging.info("[TCADP] Document parsing completed!")
                                logging.info(f"[TCADP] Task ID: {data_dict.get('TaskId')}")
                                logging.info(f"[TCADP] Success pages: {data_dict.get('SuccessPageNum')}")
                                logging.info(f"[TCADP] Failed pages: {data_dict.get('FailPageNum')}")

                                # Print failed page information
                                failed_pages = data_dict.get("FailedPages", [])
                                if failed_pages:
                                    logging.warning("[TCADP] Failed parsing pages:")
                                    for page in failed_pages:
                                        logging.warning(f"[TCADP]   Page number: {page.get('PageNumber')}, Error: {page.get('ErrorMsg')}")

                                # Check if there is a download link
                                download_url = data_dict.get("DocumentRecognizeResultUrl")
                                if download_url:
                                    logging.info(f"[TCADP] Got download link: {download_url}")
                                else:
                                    logging.warning("[TCADP] No download link obtained")

                                break  # Found final result, exit loop
                            else:
                                # Print progress information
                                progress = data_dict.get("Progress", "0")
                                logging.info(f"[TCADP] Progress: {progress}%")
                        except json.JSONDecodeError as e:
                            logging.error(f"[TCADP] Failed to parse JSON data: {e}")
                            logging.error(f"[TCADP] Raw data: {event.get('data')}")
                            continue
                    else:
                        logging.info(f"[TCADP] Event without data: {event}")
            else:  # Non-streaming response
                logging.info("[TCADP] Detected non-streaming response")
                if hasattr(resp, 'data') and resp.data:
                    try:
                        data_dict = json.loads(resp.data)
                        parser_result = data_dict
                        logging.info(f"[TCADP] JSON parsing successful: {parser_result}")
                    except json.JSONDecodeError as e:
                        logging.error(f"[TCADP] JSON parsing failed: {e}")
                        return None
                else:
                    logging.error("[TCADP] No data in response")
                    return None

            return parser_result

        except TencentCloudSDKException as err:
            logging.error(f"[TCADP] Tencent Cloud SDK error: {err}")
            return None
        except Exception as e:
            logging.error(f"[TCADP] Unknown error: {e}")
            logging.error(f"[TCADP] Error stack trace: {traceback.format_exc()}")
            return None

    def download_result_file(self, download_url, output_dir, local_cache_dir=None):
        """Download parsing result file"""
        if not download_url:
            logging.warning("[TCADP] No downloadable result file")
            return None

        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tcadp_result_{timestamp}.zip"
            file_path = os.path.join(output_dir, filename)

            with requests.get(download_url, stream=True) as response:
                response.raise_for_status()
                with open(file_path, "wb") as f:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, f)

            logging.info(f"[TCADP] Document parsing result downloaded to: {os.path.basename(file_path)}")
            return file_path

        except Exception as e:
            logging.error(f"[TCADP] Failed to download file: {e}")
            try:
                if "file_path" in locals() and os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception:
                pass

            # Try to find local file in cache directory
            if local_cache_dir:
                logging.info(f"[TCADP] Trying to find local file in cache directory: {local_cache_dir}")
                import re
                # Extract filename from download URL (without .zip extension)
                filename_match = re.search(r'/([^/]+)\.zip', download_url)
                if filename_match:
                    base_filename = filename_match.group(1)
                    local_zip_path = os.path.join(local_cache_dir, f"{base_filename}.zip")
                    logging.info(f"[TCADP] Looking for file: {local_zip_path}")
                    # List directory contents for debugging
                    try:
                        dir_contents = os.listdir(local_cache_dir)
                        zip_files = [f for f in dir_contents if f.endswith('.zip')]
                        logging.info(f"[TCADP] Cache directory contains {len(zip_files)} zip files")
                        if zip_files:
                            logging.info(f"[TCADP] First few zip files: {zip_files[:5]}")
                    except Exception as e:
                        logging.warning(f"[TCADP] Could not list cache directory: {e}")
                    if os.path.exists(local_zip_path):
                        logging.info(f"[TCADP] Found local ZIP file in cache: {local_zip_path}")
                        return local_zip_path
                    else:
                        logging.warning(f"[TCADP] Local ZIP file not found in cache: {local_zip_path}")
                        # Try case-insensitive search
                        for fname in os.listdir(local_cache_dir) if os.path.isdir(local_cache_dir) else []:
                            if fname.lower() == f"{base_filename}.zip".lower():
                                local_zip_path = os.path.join(local_cache_dir, fname)
                                logging.info(f"[TCADP] Found file with different case: {local_zip_path}")
                                return local_zip_path
            return None


class TCADPParser(RAGFlowPdfParser):
    def __init__(self, secret_id: str = None, secret_key: str = None, region: str = "ap-guangzhou",
                 table_result_type: str = None, markdown_image_response_type: str = None):
        super().__init__()

        # First initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # Log received parameters
        self.logger.info(f"[TCADP] Initializing with parameters - table_result_type: {table_result_type}, markdown_image_response_type: {markdown_image_response_type}")

        # Priority: read configuration from RAGFlow configuration system (service_conf.yaml)
        try:
            tcadp_parser = get_base_config("tcadp_config", {})
            if isinstance(tcadp_parser, dict) and tcadp_parser:
                self.secret_id = secret_id or tcadp_parser.get("secret_id")
                self.secret_key = secret_key or tcadp_parser.get("secret_key")
                self.region = region or tcadp_parser.get("region", "ap-guangzhou")
                # Set table_result_type and markdown_image_response_type from config or parameters
                self.table_result_type = table_result_type if table_result_type is not None else tcadp_parser.get("table_result_type", "1")
                self.markdown_image_response_type = markdown_image_response_type if markdown_image_response_type is not None else tcadp_parser.get("markdown_image_response_type", "1")
                # Add local_cache_dir configuration
                self.local_cache_dir = tcadp_parser.get("local_cache_dir", "/home/admin/rag")

            else:
                self.logger.error("[TCADP] Please configure tcadp_config in service_conf.yaml first")
                # If config file is empty, use provided parameters or defaults
                self.secret_id = secret_id
                self.secret_key = secret_key
                self.region = region or "ap-guangzhou"
                self.table_result_type = table_result_type if table_result_type is not None else "1"
                self.markdown_image_response_type = markdown_image_response_type if markdown_image_response_type is not None else "1"
                self.local_cache_dir = "/home/admin/rag"  # Default local cache directory

        except ImportError:
            self.logger.info("[TCADP] Configuration module import failed")
            # If config file is not available, use provided parameters or defaults
            self.secret_id = secret_id
            self.secret_key = secret_key
            self.region = region or "ap-guangzhou"
            self.table_result_type = table_result_type if table_result_type is not None else "1"
            self.markdown_image_response_type = markdown_image_response_type if markdown_image_response_type is not None else "1"
            self.local_cache_dir = "/home/admin/rag"  # Default local cache directory

        # Log final values
        self.logger.info(f"[TCADP] Final values - table_result_type: {self.table_result_type}, markdown_image_response_type: {self.markdown_image_response_type}, local_cache_dir: {self.local_cache_dir}")

        if not self.secret_id or not self.secret_key:
            raise ValueError("[TCADP] Please set Tencent Cloud API keys, configure tcadp_config in service_conf.yaml")

    @staticmethod
    def _is_zipinfo_symlink(member: zipfile.ZipInfo) -> bool:
        return (member.external_attr >> 16) & 0o170000 == 0o120000

    def check_installation(self) -> bool:
        """Check if Tencent Cloud API configuration is correct"""
        try:
            # Check necessary configuration parameters
            if not self.secret_id or not self.secret_key:
                self.logger.error("[TCADP] Tencent Cloud API configuration incomplete")
                return False

            # Try to create client to verify configuration
            TencentCloudAPIClient(self.secret_id, self.secret_key, self.region)
            self.logger.info("[TCADP] Tencent Cloud API configuration check passed")
            return True
        except Exception as e:
            self.logger.error(f"[TCADP] Tencent Cloud API configuration check failed: {e}")
            return False

    def _file_to_base64(self, file_path: str, binary: bytes = None) -> str:
        """Convert file to Base64 format"""

        if binary:
            # If binary data is directly available, convert directly
            return base64.b64encode(binary).decode('utf-8')
        else:
            # Read from file path and convert
            with open(file_path, 'rb') as f:
                file_data = f.read()
                return base64.b64encode(file_data).decode('utf-8')

    def _extract_content_from_zip(self, zip_path: str) -> list[dict[str, Any]]:
        """Extract parsing results from downloaded ZIP file"""
        results = []
        # Store images extracted from ZIP: name -> PIL Image
        zip_images = {}

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_file:
                members = zip_file.infolist()
                self.logger.info(f"[TCADP] ZIP file contains {len(members)} entries")

                # First pass: collect all images from the ZIP
                for member in members:
                    name = member.filename.replace("\\", "/")
                    # Check if this is an image file
                    if name.startswith("images/") or name.startswith("image/"):
                        img_ext = name.split('.')[-1].lower() if '.' in name else ''
                        if img_ext in ('png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'):
                            try:
                                with zip_file.open(member) as img_f:
                                    img_data = img_f.read()
                                    img = Image.open(BytesIO(img_data))
                                    img_name = os.path.basename(name)
                                    zip_images[img_name] = img
                                    zip_images[name] = img  # Also store full path
                                    self.logger.debug(f"[TCADP] Loaded image from ZIP: {name} ({len(img_data)} bytes)")
                            except Exception as e:
                                self.logger.warning(f"[TCADP] Failed to load image {name}: {e}")

                self.logger.info(f"[TCADP] Loaded {len(set(k for k in zip_images.keys() if not k.startswith('images/')))} unique images from ZIP")

                for member in members:
                    name = member.filename.replace("\\", "/")
                    self.logger.debug(f"[TCADP] Processing ZIP entry: {name}")

                    if member.is_dir():
                        self.logger.debug(f"[TCADP] Skipping directory: {name}")
                        continue

                    if member.flag_bits & 0x1:
                        raise RuntimeError(f"[TCADP] Encrypted zip entry not supported: {member.filename}")
                    if self._is_zipinfo_symlink(member):
                        raise RuntimeError(f"[TCADP] Symlink zip entry not supported: {member.filename}")
                    if name.startswith("/") or name.startswith("//") or re.match(r"^[A-Za-z]:", name):
                        raise RuntimeError(f"[TCADP] Unsafe zip path (absolute): {member.filename}")
                    parts = [p for p in name.split("/") if p not in ("", ".")]
                    if any(p == ".." for p in parts):
                        raise RuntimeError(f"[TCADP] Unsafe zip path (traversal): {member.filename}")

                    # Skip images and other non-text files in main processing
                    if name.startswith("images/") or name.startswith("image/"):
                        continue

                    if not (name.endswith(".json") or name.endswith(".md")):
                        self.logger.debug(f"[TCADP] Skipping non-JSON/MD file: {name}")
                        continue

                    try:
                        with zip_file.open(member) as f:
                            if name.endswith(".json"):
                                data = json.load(f)
                                self.logger.debug(f"[TCADP] JSON file {name}: type={type(data)}, keys={list(data.keys()) if isinstance(data, dict) else 'N/A'}")

                                if isinstance(data, dict):
                                    # Check if this is page-level OCR result (has PageNumber and Elements)
                                    if 'PageNumber' in data and 'Elements' in data:
                                        page_num = data.get('PageNumber', 1)
                                        elements = data.get('Elements', [])
                                        self.logger.info(f"[TCADP] Page {page_num}: {len(elements)} elements")

                                        # Merge consecutive text elements on the same page
                                        merged_blocks = self._merge_page_elements(elements, zip_images, zip_file)

                                        for block in merged_blocks:
                                            result_block = {
                                                'type': block.get('type', 'paragraph'),
                                                'content': block.get('content', ''),
                                                'page_number': page_num,
                                                'level': block.get('level', 0),
                                            }
                                            # If block has image, add it for RAGFlow to process
                                            if block.get('image'):
                                                result_block['image'] = block['image']
                                            results.append(result_block)
                                    elif 'pages' in data and isinstance(data['pages'], list):
                                        # Nested pages format
                                        for i, page in enumerate(data['pages']):
                                            if isinstance(page, dict):
                                                page['page_number'] = page.get('page_number', i + 1)
                                            results.append(page)
                                    elif 'content' in data or 'text' in data or 'Text' in data:
                                        # Single content object
                                        results.append(data)
                                    else:
                                        results.append(data)
                                elif isinstance(data, list):
                                    # List of content blocks
                                    results.extend(data)
                                else:
                                    results.append({"type": "text", "content": str(data), "file": name})
                            else:
                                content = f.read().decode("utf-8")
                                self.logger.debug(f"[TCADP] MD file {name}: content_length={len(content)}")
                                # For md_full.md, split into sections by headers
                                if 'md_full' in name:
                                    sections = self._split_markdown_by_headers(content, name)
                                    results.extend(sections)
                                else:
                                    results.append({"type": "text", "content": content, "file": name, "page_number": 0})
                    except Exception as e:
                        self.logger.error(f"[TCADP] Failed to process {name}: {e}")
                        continue

        except Exception as e:
            self.logger.error(f"[TCADP] Failed to extract ZIP file content: {e}")

        self.logger.info(f"[TCADP] Extracted {len(results)} total content blocks")
        # Log first few blocks for debugging
        for i, block in enumerate(results[:5]):
            if isinstance(block, dict):
                has_img = 'image' in block
                content_len = len(block.get('content', ''))
                self.logger.info(f"[TCADP] Block {i}: type={block.get('type', 'N/A')}, page={block.get('page_number', 'N/A')}, content_len={content_len}, has_image={has_img}")
            else:
                self.logger.info(f"[TCADP] Block {i}: type={type(block)}, value={str(block)[:100]}")
        return results

    def _merge_page_elements(self, elements: list, zip_images: dict, zip_file) -> list:
        """Merge consecutive text elements on the same page into larger blocks

        Merge strategy:
        - Group consecutive non-table text elements
        - Keep table and image elements separate
        - Respect title/header boundaries
        - Handle images by storing PIL Image objects for RAGFlow processing
        """
        if not elements:
            return []

        merged = []
        current_block = {
            'content_parts': [],
            'type': 'paragraph',
            'level': 0,
            'image': None  # Store PIL Image object
        }

        # Element types that should start a new block
        break_types = {'table', 'image', 'figure'}
        # Element types that are headers/titles (higher priority)
        header_types = {'title', 'header', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}

        for elem in elements:
            if not isinstance(elem, dict):
                continue

            elem_text = elem.get('Text', '').strip()
            elem_type = elem.get('Type', 'paragraph').lower()
            image_path = elem.get('ImagePath', '')
            inset_image = elem.get('InsetImageName', '')
            elem_level = elem.get('Level', 0)

            # Skip empty text elements without images
            if not elem_text and not image_path and not inset_image:
                continue

            # Get associated image if any
            elem_image = None
            if image_path:
                elem_image = self._get_image_from_zip(image_path, zip_images, zip_file)
            if not elem_image and inset_image:
                elem_image = self._get_image_from_zip(inset_image, zip_images, zip_file)

            # Decide whether to start a new block or append to current
            should_break = False

            # Break if current block has content and:
            # 1. This is a table/image/figure
            # 2. This is a header/title (start new section)
            # 3. Current block already has significant content (>300 chars)
            # 4. Current block already has an image
            # 5. This element has an image and current block has content

            if current_block['content_parts']:
                if elem_type in break_types:
                    should_break = True
                elif elem_type in header_types:
                    should_break = True
                elif current_block['image'] is not None:
                    should_break = True
                elif elem_image is not None:
                    should_break = True
                elif len("\n".join(current_block['content_parts'])) > 400:
                    should_break = True

            if should_break:
                # Save current block
                if current_block['content_parts'] or current_block['image']:
                    block_data = {
                        'type': current_block['type'],
                        'content': "\n".join(current_block['content_parts']),
                        'level': current_block['level'],
                    }
                    if current_block['image']:
                        block_data['image'] = current_block['image']
                    merged.append(block_data)
                # Start new block
                current_block = {
                    'content_parts': [elem_text] if elem_text else [],
                    'type': elem_type,
                    'level': elem_level,
                    'image': elem_image
                }
            else:
                # Append to current block
                if elem_text:
                    current_block['content_parts'].append(elem_text)
                # Update block type if this is a header
                if elem_type in header_types:
                    current_block['type'] = elem_type
                    current_block['level'] = elem_level
                # Store image if present (shouldn't happen often due to should_break logic)
                if elem_image:
                    current_block['image'] = elem_image

        # Don't forget the last block
        if current_block['content_parts'] or current_block['image']:
            block_data = {
                'type': current_block['type'],
                'content': "\n".join(current_block['content_parts']),
                'level': current_block['level'],
            }
            if current_block['image']:
                block_data['image'] = current_block['image']
            merged.append(block_data)

        self.logger.info(f"[TCADP] Merged {len(elements)} elements into {len(merged)} blocks")
        return merged

    def _get_image_from_zip(self, image_path: str, zip_images: dict, zip_file) -> Image.Image | None:
        """Get PIL Image from ZIP by path

        Args:
            image_path: Path to image (e.g., 'images/xxx.png' or just 'xxx.png')
            zip_images: Cache dict of loaded images
            zip_file: ZIP file handle

        Returns:
            PIL Image object or None if not found
        """
        if not image_path:
            return None

        # Normalize path
        image_name = os.path.basename(image_path)

        # Try to get image from cache
        if image_name in zip_images:
            return zip_images[image_name]
        if image_path in zip_images:
            return zip_images[image_path]

        # Try to load from ZIP dynamically
        try:
            # Handle both 'images/xxx.png' and just 'xxx.png'
            possible_paths = [
                image_path,
                f"images/{image_name}",
                f"image/{image_name}",
            ]
            for path in possible_paths:
                try:
                    with zip_file.open(path) as img_f:
                        img_data = img_f.read()
                        img = Image.open(BytesIO(img_data))
                        zip_images[image_name] = img
                        zip_images[path] = img
                        self.logger.debug(f"[TCADP] Dynamically loaded image: {path}")
                        return img
                except KeyError:
                    continue
        except Exception as e:
            self.logger.warning(f"[TCADP] Could not load image {image_path}: {e}")

        return None

    def _split_markdown_by_headers(self, content: str, filename: str) -> list[dict[str, Any]]:
        """Split markdown content by headers into sections"""
        sections = []
        # Split by markdown headers (## or #)
        # Pattern to match headers and their content
        pattern = r'(^#{1,6}\s+.+$)(.*?)(?=^#{1,6}\s+|$)'
        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)

        if matches:
            for header, body in matches:
                section_content = (header + body).strip()
                if section_content:
                    sections.append({
                        "type": "text",
                        "content": section_content,
                        "file": filename,
                        "page_number": 0
                    })
        else:
            # No headers found, treat as single section
            if content.strip():
                sections.append({
                    "type": "text",
                    "content": content,
                    "file": filename,
                    "page_number": 0
                })

        self.logger.info(f"[TCADP] Split markdown into {len(sections)} sections")
        return sections

    def _parse_content_to_sections(self, content_data: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """Convert parsing results to sections format"""
        sections = []

        self.logger.info(f"[TCADP] Starting to parse {len(content_data)} content blocks to sections")

        for idx, item in enumerate(content_data):
            # Support both old and new field names
            content_type = item.get("type", item.get("Type", "text")).lower()
            content = item.get("content", item.get("text", item.get("Text", "")))
            page_num = item.get("page_number", item.get("page", item.get("PageNumber", 0)))

            self.logger.debug(f"[TCADP] Block {idx}: type={content_type}, page={page_num}, content_length={len(content) if content else 0}")

            if not content:
                self.logger.debug(f"[TCADP] Block {idx}: skipping due to empty content")
                continue

            # Process based on content type
            if content_type in ("text", "paragraph", "title", "header", "footer"):
                section_text = content
            elif content_type == "table":
                # Handle table content
                table_data = item.get("table_data", {})
                if isinstance(table_data, dict):
                    # Convert table data to text
                    rows = table_data.get("rows", [])
                    section_text = "\n".join([" | ".join(row) for row in rows])
                else:
                    section_text = str(table_data)
            elif content_type == "image":
                # Handle image content
                caption = item.get("caption", "")
                section_text = f"[Image] {caption}" if caption else "[Image]"
            elif content_type == "equation":
                # Handle equation content
                section_text = f"$${content}$$"
            else:
                # Unknown type, treat as text
                self.logger.debug(f"[TCADP] Block {idx}: unknown type '{content_type}', treating as text")
                section_text = content

            if section_text.strip():
                # Generate position tag with page number
                position_tag = f"@@{page_num}\t0.0\t1000.0\t0.0\t100.0##" if page_num else "@@1\t0.0\t1000.0\t0.0\t100.0##"
                sections.append((section_text, position_tag))
                self.logger.debug(f"[TCADP] Block {idx}: added as section, length={len(section_text)}")
            else:
                self.logger.debug(f"[TCADP] Block {idx}: skipping due to empty section_text after processing")

        self.logger.info(f"[TCADP] Parsed {len(sections)} sections from {len(content_data)} content blocks")
        return sections

    def _parse_content_to_tables(self, content_data: list[dict[str, Any]]) -> list:
        """Convert parsing results to tables format"""
        tables = []

        for item in content_data:
            # Support both 'type' and 'Type' fields
            content_type = item.get("type", item.get("Type", "")).lower()
            if content_type == "table":
                table_data = item.get("table_data", item.get("TableData", {}))
                if isinstance(table_data, dict):
                    rows = table_data.get("rows", table_data.get("Rows", []))
                    if rows:
                        # Convert to table format
                        table_html = "<table>\n"
                        for i, row in enumerate(rows):
                            table_html += "  <tr>\n"
                            for cell in row:
                                tag = "th" if i == 0 else "td"
                                table_html += f"    <{tag}>{cell}</{tag}>\n"
                            table_html += "  </tr>\n"
                        table_html += "</table>"
                        tables.append(table_html)

        return tables

    def parse_pdf(
        self,
        filepath: str | PathLike[str],
        binary: BytesIO | bytes,
        callback: Optional[Callable] = None,
        *,
        output_dir: Optional[str] = None,
        file_type: str = "PDF",
        file_start_page: Optional[int] = 1,
        file_end_page: Optional[int] = 1000,
        delete_output: Optional[bool] = True,
        max_retries: Optional[int] = 1,
        local_zip_path: Optional[str] = None,
    ) -> tuple:
        """Parse PDF document"""

        temp_file = None
        created_tmp_dir = False
        out_dir = None

        try:
            if local_zip_path:
                self.logger.info(f"[TCADP] Using local ZIP file: {local_zip_path}")
                if not os.path.exists(local_zip_path):
                    if callback:
                        callback(-1, f"[TCADP] Local ZIP file does not exist: {local_zip_path}")
                    raise FileNotFoundError(f"[TCADP] Local ZIP file does not exist: {local_zip_path}")

                if callback:
                    callback(0.6, f"[TCADP] Using local ZIP file: {os.path.basename(local_zip_path)}")

                zip_path = local_zip_path
            else:
                temp_file, file_path = handle_input_file(self, filepath, binary, callback)
                file_base64 = convert_to_base64(self, file_path, binary, callback)
                client = TencentCloudAPIClient(self.secret_id, self.secret_key, self.region)
                result = call_tencent_cloud_api(self, client, file_type, file_base64, file_start_page, file_end_page, callback, max_retries)
                download_url = result.get("DocumentRecognizeResultUrl")
                if not download_url:
                    if callback:
                        callback(-1, "[TCADP] No parsing result download link obtained")
                    raise RuntimeError("[TCADP] No parsing result download link obtained")

                if callback:
                    callback(0.6, f"[TCADP] Parsing result download link: {download_url}")

                out_dir, created_tmp_dir = setup_output_dir(output_dir)
                zip_path = download_result_file(client, download_url, out_dir, callback, self.local_cache_dir)
                if not zip_path:
                    if callback:
                        callback(-1, "[TCADP] Failed to download parsing result")
                    raise RuntimeError("[TCADP] Failed to download parsing result")

            content_data = self._extract_content_from_zip(zip_path)
            self.logger.info(f"[TCADP] Extracted {len(content_data)} content blocks")

            if callback:
                callback(0.9, f"[TCADP] Extracted {len(content_data)} content blocks")

            sections = self._parse_content_to_sections(content_data)
            tables = self._parse_content_to_tables(content_data)

            self.logger.info(f"[TCADP] Parsing completed: {len(sections)} sections, {len(tables)} tables")

            if callback:
                callback(1.0, f"[TCADP] Parsing completed: {len(sections)} sections, {len(tables)} tables")

            return sections, tables

        finally:
            cleanup_temp_files(temp_file, delete_output, created_tmp_dir, out_dir)


def handle_input_file(parser, filepath, binary, callback):
    if binary:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.write(binary)
        temp_file.close()
        file_path = temp_file.name
        parser.logger.info(f"[TCADP] Received binary PDF -> {os.path.basename(file_path)}")
        if callback:
            callback(0.1, f"[TCADP] Received binary PDF -> {os.path.basename(file_path)}")
    else:
        file_path = str(filepath)
        if not os.path.exists(file_path):
            if callback:
                callback(-1, f"[TCADP] PDF file does not exist: {file_path}")
            raise FileNotFoundError(f"[TCADP] PDF file does not exist: {file_path}")
        temp_file = None
    return temp_file, file_path


def convert_to_base64(parser, filepath, binary, callback):
    if callback:
        callback(0.2, "[TCADP] Converting file to Base64 format")

    file_base64 = parser._file_to_base64(filepath, binary)
    if callback:
        callback(0.25, f"[TCADP] File converted to Base64, size: {len(file_base64)} characters")
    return file_base64


def call_tencent_cloud_api(parser, client, file_type, file_base64, file_start_page, file_end_page, callback, max_retries):
    if callback:
        callback(0.3, "[TCADP] Starting to call Tencent Cloud document parsing API")

    result = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                parser.logger.info(f"[TCADP] Retry attempt {attempt + 1}")
                if callback:
                    callback(0.3 + attempt * 0.1, f"[TCADP] Retry attempt {attempt + 1}")
                time.sleep(2 ** attempt)

            config = {
                "TableResultType": parser.table_result_type,
                "MarkdownImageResponseType": parser.markdown_image_response_type
            }

            parser.logger.info(f"[TCADP] API request config - TableResultType: {parser.table_result_type}, MarkdownImageResponseType: {parser.markdown_image_response_type}")

            result = client.reconstruct_document_sse(
                file_type=file_type,
                file_base64=file_base64,
                file_start_page=file_start_page,
                file_end_page=file_end_page,
                config=config
            )

            if result:
                parser.logger.info(f"[TCADP] Attempt {attempt + 1} successful")
                break
            else:
                parser.logger.warning(f"[TCADP] Attempt {attempt + 1} failed, result is None")

        except Exception as e:
            parser.logger.error(f"[TCADP] Attempt {attempt + 1} exception: {e}")
            if attempt == max_retries - 1:
                raise

    if not result:
        error_msg = f"[TCADP] Document parsing failed, retried {max_retries} times"
        parser.logger.error(error_msg)
        if callback:
            callback(-1, error_msg)
        raise RuntimeError(error_msg)

    return result


def setup_output_dir(output_dir):
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        created_tmp_dir = False
    else:
        out_dir = Path(tempfile.mkdtemp(prefix="adp_pdf_"))
        created_tmp_dir = True
    return out_dir, created_tmp_dir


def download_result_file(client, download_url, output_dir, callback, local_cache_dir=None):
    zip_path = client.download_result_file(download_url, output_dir, local_cache_dir)
    if not zip_path:
        return None

    if callback:
        zip_filename = os.path.basename(zip_path)
        callback(0.8, f"[TCADP] Parsing result downloaded: {zip_filename}")

    return zip_path


def cleanup_temp_files(temp_file, delete_output, created_tmp_dir, out_dir):
    if temp_file and os.path.exists(temp_file.name):
        try:
            os.unlink(temp_file.name)
        except Exception:
            pass

    if delete_output and created_tmp_dir and out_dir.exists():
        try:
            shutil.rmtree(out_dir)
        except Exception:
            pass


if __name__ == "__main__":
    # Test ADP parser
    parser = TCADPParser()
    print("ADP available:", parser.check_installation())

    # Test parsing
    filepath = ""
    if filepath and os.path.exists(filepath):
        with open(filepath, "rb") as file:
            sections, tables = parser.parse_pdf(filepath=filepath, binary=file.read())
            print(f"Parsing result: {len(sections)} sections, {len(tables)} tables")
            for i, (section, tag) in enumerate(sections[:3]):  # Only print first 3
                print(f"Section {i + 1}: {section[:100]}...")
