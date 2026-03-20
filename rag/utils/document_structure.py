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

import re
from typing import List, Dict, Tuple, Optional
import logging


class DocumentStructureElement:
    """文档结构元素"""
    def __init__(self, element_type: str, content: str, level: int = 0, 
                 start_pos: int = 0, end_pos: int = 0):
        self.type = element_type  # 'heading', 'paragraph', 'list', 'table', 'image', 'code'
        self.content = content
        self.level = level  # 层级，主要用于标题
        self.start_pos = start_pos  # 在原文中的起始位置
        self.end_pos = end_pos  # 在原文中的结束位置
        self.children: List['DocumentStructureElement'] = []


class ProtectedContentSpan:
    """受保护的内容片段"""
    def __init__(self, start: int, end: int, content_type: str):
        self.start = start
        self.end = end
        self.type = content_type  # 'table', 'image', 'code', 'math', 'link'


class DocumentStructureAnalyzer:
    """文档结构分析器"""
    
    def __init__(self):
        # 定义各种受保护内容的正则模式
        self.protected_patterns = {
            'math_block': re.compile(r'\$\$.*?\$\$', re.DOTALL),
            'image': re.compile(r'!\[[^\]]*\]\([^)]+\)'),
            'link': re.compile(r'\[[^\]]*\]\([^)]+\)'),
            'table': re.compile(r'(?:\|[^\n|]+)+\|[\r\n]+(?:\|\s*:?-+:?\s*\|[\r\n]+)?(?:\|[^\n|]+\|[\r\n]+)+'),
            'code_block': re.compile(r'```[\s\S]*?```'),
            # HTML 表格 (mineru 输出格式)
            'html_table': re.compile(r'<table>[\s\S]*?</table>', re.DOTALL),
            'inline_code': re.compile(r'`[^`]+`'),
            # 数学公式：$...\%$ 或 $...$ (mineru输出格式如 $3 . 1 \%$)，支持多行
            # 关键：不要让 [^$] 匹配 \%，否则 \%$ 会被分开
            # 使用负向前瞻确保 \ 不被 [^$] 匹配
            'math_inline': re.compile(r'\$[^\$]*(?:\\.[^\$]*)*(?:\%$|\$)', re.DOTALL)
        }
        
        # 标题模式
        self.heading_patterns = [
            (re.compile(r'^#{1,6}\s+(.+)$', re.MULTILINE), range(1, 7)),  # Markdown标题
            (re.compile(r'^(.+?)\n[-=]{3,}$', re.MULTILINE), [1, 2]),      # Setext标题
        ]

    def analyze_structure(self, markdown_content: str) -> Tuple[List[DocumentStructureElement], List[ProtectedContentSpan]]:
        """
        分析文档结构，返回结构元素和受保护内容片段
        
        Args:
            markdown_content: Markdown格式的文档内容
            
        Returns:
            Tuple[List[DocumentStructureElement], List[ProtectedContentSpan]]: 
            文档结构元素列表和受保护内容片段列表
        """
        structure_elements = []
        protected_spans = []
        
        # 1. 找出所有受保护的内容片段
        protected_spans = self._find_protected_spans(markdown_content)
        
        # 2. 解析标题结构
        headings = self._parse_headings(markdown_content)
        
        # 3. 按结构分割内容
        structure_elements = self._build_structure_elements(markdown_content, headings, protected_spans)
        
        return structure_elements, protected_spans

    def _find_protected_spans(self, text: str) -> List[ProtectedContentSpan]:
        """找出所有不应被分割的受保护内容片段"""
        spans = []
        
        for content_type, pattern in self.protected_patterns.items():
            for match in pattern.finditer(text):
                spans.append(ProtectedContentSpan(
                    start=match.start(),
                    end=match.end(),
                    content_type=content_type
                ))
        
        # 按起始位置排序并去除重叠
        spans.sort(key=lambda x: x.start)
        non_overlapping_spans = []
        
        last_end = 0
        for span in spans:
            if span.start >= last_end:
                non_overlapping_spans.append(span)
                last_end = span.end
                
        return non_overlapping_spans

    def _parse_headings(self, text: str) -> List[Tuple[int, str, int, int]]:
        """解析文档中的标题
        
        Returns:
            List[Tuple[level, title_text, start_pos, end_pos]]
        """
        headings = []
        
        for pattern, levels in self.heading_patterns:
            for match in pattern.finditer(text):
                if pattern.pattern.startswith(r'^#{1,6}'):
                    # Markdown标题
                    hashes = match.group(0).split()[0]
                    level = len(hashes)
                    title_text = match.group(1)
                else:
                    # Setext标题
                    title_text = match.group(1)
                    underline = match.group(0)[len(title_text):].strip()
                    level = 1 if '=' in underline else 2
                    
                if level in levels:
                    headings.append((
                        level, 
                        title_text.strip(), 
                        match.start(), 
                        match.end()
                    ))
        
        # 按位置排序
        headings.sort(key=lambda x: x[2])
        return headings

    def _build_structure_elements(self, text: str, headings: List, protected_spans: List[ProtectedContentSpan]) -> List[DocumentStructureElement]:
        """构建文档结构元素"""
        elements = []
        
        # 如果没有标题，将整个文档作为一个段落
        if not headings:
            return [DocumentStructureElement(
                element_type='paragraph',
                content=text.strip(),
                level=0,
                start_pos=0,
                end_pos=len(text)
            )]
        
        # 按标题分割文档
        last_end = 0
        for level, title_text, start_pos, end_pos in headings:
            # 添加标题前的内容作为段落
            if start_pos > last_end:
                paragraph_content = text[last_end:start_pos].strip()
                if paragraph_content:
                    elements.append(DocumentStructureElement(
                        element_type='paragraph',
                        content=paragraph_content,
                        level=0,
                        start_pos=last_end,
                        end_pos=start_pos
                    ))
            
            # 添加标题
            elements.append(DocumentStructureElement(
                element_type='heading',
                content=title_text,
                level=level,
                start_pos=start_pos,
                end_pos=end_pos
            ))
            
            last_end = end_pos
        
        # 添加最后一个标题后的内容
        if last_end < len(text):
            remaining_content = text[last_end:].strip()
            if remaining_content:
                elements.append(DocumentStructureElement(
                    element_type='paragraph',
                    content=remaining_content,
                    level=0,
                    start_pos=last_end,
                    end_pos=len(text)
                ))
        
        return elements

    def smart_split_text(self, text: str, max_chunk_size: int = 512,
                        separators: List[str] = None) -> List[str]:
        """
        智能分割 Markdown 文档

        核心策略：按标题层级分段，在大段内部按段落/句子分 chunk

        处理流程：
        1. 预处理：清理噪声、合并跨行内容
        2. 识别标题层级，建立文档结构树
        3. 按标题划分大段落
        4. 每个大段落如果超过 max_chunk_size，再按段落分割
        5. 合并过短的 chunks

        Args:
            text: 要分割的 Markdown 文本
            max_chunk_size: 最大 chunk 大小（字节数）
            separators: 分割符列表（备用）

        Returns:
            List[str]: 分割后的文本块列表
        """
        import logging
        logger = logging.getLogger(__name__)

        if not text or not text.strip():
            return []

        # 预处理
        text = self._preprocess_text(text)

        # 解析文档结构
        elements, protected_spans = self.analyze_structure(text)

        # 找出所有标题
        headings = self._find_all_headings(text)

        if not headings:
            # 没有标题时，按段落分割
            logger.info("无标题，使用段落分割策略")
            return self._split_by_paragraphs(text, max_chunk_size, protected_spans)

        # 按标题分割大段落
        sections = self._split_by_headings(text, headings)

        # 合并和分割每个大段落
        chunks = []
        for section_text, heading_level in sections:
            if not section_text.strip():
                continue

            section_size = len(section_text.encode('utf-8'))

            if section_size <= max_chunk_size:
                # 段落足够小，直接保留
                if section_text.strip():
                    chunks.append(section_text.strip())
            else:
                # 段落太大，需要进一步分割
                sub_chunks = self._split_large_section(
                    section_text, max_chunk_size, protected_spans
                )
                chunks.extend(sub_chunks)

        # 合并过短的 chunks
        chunks = self._merge_short_chunks(chunks, max_chunk_size)

        logger.info(f"Markdown 分割完成: {len(chunks)} 个 chunks")
        return chunks

    def _find_all_headings(self, text: str) -> List[Tuple[int, str, int, int, int]]:
        """找出所有标题

        Returns:
            List[Tuple[level, title, start, end, heading_end]]
            level: 标题级别 1-6
            title: 标题文本
            start: 标题开始位置
            end: 标题行结束位置（包含换行）
            heading_end: 标题内容结束位置（下一个非空行开始）
        """
        headings = []
        lines = text.split('\n')

        current_pos = 0
        for i, line in enumerate(lines):
            # 检查是否是 Markdown 标题
            match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if match:
                level = len(match.group(1))
                title = match.group(2)
                heading_line_start = current_pos
                heading_line_end = current_pos + len(line) + 1  # +1 for \n
                # 找到这个标题对应的内容结束位置（下一个同级或更高级标题之前）
                content_end = self._find_section_end(lines, i, level, current_pos)

                headings.append((
                    level,
                    title,
                    heading_line_start,
                    heading_line_end,
                    content_end
                ))

            current_pos += len(line) + 1  # +1 for \n
        # 按位置排序
        headings.sort(key=lambda x: x[2])
        return headings

    def _find_section_end(self, lines: List[str], heading_idx: int,
                         heading_level: int, heading_start_pos: int) -> int:
        """找到标题对应内容的结束位置"""
        content_end = heading_start_pos + len(lines[heading_idx]) + 1  # 从标题行结束后开始

        # 跳过标题后的空行
        for j in range(heading_idx + 1, len(lines)):
            if lines[j].strip():
                break
            content_end += len(lines[j]) + 1

        # 找到下一个同级或更高级标题
        for j in range(heading_idx + 1, len(lines)):
            line = lines[j].strip()
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match:
                next_level = len(match.group(1))
                if next_level <= heading_level:
                    # 找到同级或更高级标题，内容到这里结束
                    # content_end 是下一个标题的开始位置
                    break
            content_end += len(lines[j]) + 1

        return content_end

    def _split_by_headings(self, text: str, headings: List) -> List[Tuple[str, int]]:
        """按标题分割文档

        Returns:
            List[Tuple[section_text, heading_level]]
        """
        sections = []

        for i, (level, title, heading_start, heading_end, content_end) in enumerate(headings):
            # 收集标题和内容
            section_start = heading_start

            # 确定section的结束位置
            if i + 1 < len(headings):
                next_heading_start = headings[i + 1][2]
                section_end = min(content_end, next_heading_start)
            else:
                section_end = max(content_end, len(text))

            section_text = text[section_start:section_end].strip()
            if section_text:
                sections.append((section_text, level))

        return sections

    def _split_by_paragraphs(self, text: str, max_chunk_size: int,
                             protected_spans: List) -> List[str]:
        """没有标题时，按段落分割"""
        import logging
        logger = logging.getLogger(__name__)

        # 按 \n\n 分割段落
        paragraphs = re.split(r'\n\n+', text)

        chunks = []
        current_chunk = ""
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para.encode('utf-8'))

            # 如果单个段落就超过限制，需要特殊处理
            if para_size > max_chunk_size * 1.5:
                # 保存当前 chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_size = 0

                # 直接添加超长段落（受保护内容会在后续处理）
                chunks.append(para)
                continue

            if current_size + para_size + 2 <= max_chunk_size:  # +2 for \n\n
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_size += para_size + 2
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
                current_size = para_size

        if current_chunk:
            chunks.append(current_chunk.strip())

        logger.info(f"按段落分割: {len(chunks)} 个 chunks")
        return chunks

    def _split_large_section(self, text: str, max_chunk_size: int,
                            protected_spans: List) -> List[str]:
        """分割过大的段落

        策略：按段落分割，保留受保护内容完整性
        """
        import logging
        logger = logging.getLogger(__name__)

        # 先处理受保护内容的边界
        protected_ranges = [(s.start, s.end) for s in protected_spans]

        # 按段落分割
        chunks = []
        current_chunk = ""
        current_size = 0

        # 按 \n 分割行，收集段落
        lines = text.split('\n')
        current_para = []
        para_sizes = []

        for line in lines:
            line_stripped = line.strip()

            # 空行表示段落结束
            if not line_stripped:
                if current_para:
                    para_text = '\n'.join(current_para)
                    para_size = len(para_text.encode('utf-8'))
                    para_sizes.append((para_text, para_size))
                    current_para = []
                continue

            current_para.append(line)

        # 处理最后一个段落
        if current_para:
            para_text = '\n'.join(current_para)
            para_size = len(para_text.encode('utf-8'))
            para_sizes.append((para_text, para_size))

        # 合并段落成 chunks
        for para_text, para_size in para_sizes:
            # 检查段落是否与受保护内容重叠
            is_protected = False
            for p_start, p_end in protected_ranges:
                text_start = text.find(para_text)
                if text_start >= 0:
                    text_end = text_start + len(para_text)
                    if not (text_end <= p_start or text_start >= p_end):
                        is_protected = True
                        break

            if is_protected:
                # 受保护段落：先保存当前 chunk，再单独添加
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if para_text.strip():
                    chunks.append(para_text.strip())
                current_chunk = ""
                current_size = 0
            elif current_size + para_size + 2 <= max_chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para_text
                else:
                    current_chunk = para_text
                current_size += para_size + 2
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para_text
                current_size = para_size

        if current_chunk:
            chunks.append(current_chunk.strip())

        logger.debug(f"大段落分割: {len(chunks)} 个 sub-chunks")
        return chunks

    def _merge_short_chunks(self, chunks: List[str], max_chunk_size: int) -> List[str]:
        """合并过短的 chunks

        策略：
        1. 纯标题 chunk（只有标题行或标题+极少内容）必须与下一个 chunk 合并
        2. 孤立图片 chunk（只有图片引用没有上下文）必须与下一个 chunk 合并
        3. 过短 chunk（< max_chunk_size / 4）强制与下一个合并
        """
        import logging
        logger = logging.getLogger(__name__)

        if not chunks:
            return chunks

        # 阈值
        min_threshold = max_chunk_size // 4
        max_threshold = max_chunk_size * 2

        result = []
        i = 0

        while i < len(chunks):
            chunk = chunks[i]
            chunk_stripped = chunk.strip()
            if not chunk_stripped:
                i += 1
                continue

            chunk_size = len(chunk_stripped.encode('utf-8'))
            lines = chunk_stripped.split('\n')
            first_line = lines[0] if lines else ""

            # 判断是否是"无价值"的 chunk
            is_heading_only = self._is_heading_only_chunk(chunk_stripped)
            is_image_only = self._is_image_only_chunk(chunk_stripped)
            is_too_small = chunk_size < min_threshold

            # 这些情况必须与下一个 chunk 合并
            if is_heading_only or is_image_only or is_too_small:
                # 收集下一个或多个 chunk 来合并
                combined = chunk_stripped
                combined_size = chunk_size
                j = i + 1

                while j < len(chunks):
                    next_chunk = chunks[j].strip()
                    if not next_chunk:
                        j += 1
                        continue

                    next_size = len(next_chunk.encode('utf-8'))

                    # 如果合并后不超过最大阈值，或者下一个也是"无价值"的
                    next_is_heading = self._is_heading_only_chunk(next_chunk)
                    next_is_image = self._is_image_only_chunk(next_chunk)

                    if combined_size + next_size + 2 <= max_threshold:
                        combined = combined + "\n\n" + next_chunk
                        combined_size += next_size + 2
                        j += 1
                    elif next_is_heading or next_is_image:
                        # 下一个也是无价值的，强制合并
                        combined = combined + "\n\n" + next_chunk
                        combined_size += next_size + 2
                        j += 1
                    else:
                        break

                result.append(combined)
                i = j
            else:
                result.append(chunk_stripped)
                i += 1

        # 最终检查：合并仍然过小的 chunks
        final_result = []
        for chunk in result:
            chunk_stripped = chunk.strip()
            if not chunk_stripped:
                continue

            chunk_size = len(chunk_stripped.encode('utf-8'))

            # 如果太小，尝试与上一个合并
            if chunk_size < min_threshold and final_result:
                combined = final_result[-1] + "\n\n" + chunk_stripped
                if len(combined.encode('utf-8')) <= max_threshold:
                    final_result[-1] = combined
                    continue

            final_result.append(chunk_stripped)

        logger.info(f"合并短 chunks: {len(chunks)} -> {len(final_result)}")
        return final_result

    def _is_heading_only_chunk(self, chunk: str) -> bool:
        """判断是否是只有标题的 chunk（没有实质内容）"""
        lines = chunk.strip().split('\n')
        if not lines:
            return False

        first_line = lines[0].strip()

        # 第一行必须是标题
        if not first_line.startswith('#'):
            return False

        # 如果只有一行（只有标题），认为是纯标题
        if len(lines) == 1:
            return True

        # 检查其他行是否都是空的
        content_lines = [l for l in lines[1:] if l.strip()]
        if not content_lines:
            return True

        # 如果内容行 <= 2 行且总内容很短（< 300 字符），认为是纯标题
        # 这些通常是：标题 + 图注，或者标题 + 很短的引言
        total_content_len = sum(len(l) for l in content_lines)
        if len(content_lines) <= 2 and total_content_len < 300:
            return True

        return False

    def _is_image_only_chunk(self, chunk: str) -> bool:
        """判断是否只有图片引用没有上下文的 chunk"""
        lines = chunk.strip().split('\n')
        if not lines:
            return False

        # 检查是否只有图片引用行
        image_pattern = re.compile(r'^!\[[^\]]*\]\([^)]+\)')
        content_lines = [l for l in lines if l.strip()]

        if not content_lines:
            return False

        # 如果只有 1-2 行且都是图片引用，认为是孤立图片
        if len(content_lines) <= 2:
            all_images = all(image_pattern.match(l.strip()) for l in content_lines)
            if all_images:
                return True

        return False
    
    def _preprocess_text(self, text: str) -> str:
        """
        预处理文本：清理噪声、合并断行内容
        
        处理规则：
        1. 移除独立的 "Text" 行
        2. 合并跨行的数学公式（$...$ 可能被
分割）
        3. 合并跨行的图表标记（表：、图：可能单独成行）
        """
        # 先合并所有断行的数学公式
        # 匹配 $ ... \%$ 或 $ ... $ 模式，支持转义字符和换行
        text = re.sub(r'\$[^\$]*(?:\\.[^\$]*)*(?:\n[^\$]*)*(?:\%$|\$)',
                      lambda m: m.group(0).replace('\n', ' '),
                      text)
        
        lines = text.split('\n')
        result_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # 跳过独立的 "Text" 元信息标记
            if line == "Text":
                i += 1
                continue
            
            result_lines.append(lines[i])
            i += 1
        
        return '\n'.join(result_lines)
    
    def _post_process_chunks(self, chunks: List[str]) -> List[str]:
        """
        后处理chunks：过滤噪声、合并短chunk
        
        处理规则：
        1. 以标点符号/公式/表格标记开头 → 合并到上一个
        2. 以不完整句子开头（括号、引号、逗号、句号等在前一个chunk末尾）→ 合并到上一个
        3. 以不完整句子结尾（括号、引号等未闭合）→ 合并到下一个
        4. 过短chunk → 合并到上一个或下一个
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if not chunks:
            return chunks
        
        # 定义不完整的开头字符
        OPEN_BRACKETS = '（）【】[《》<>"\'‘’'

        # 中文逗号、句号等开头的应该合并到上一个
        TRAILING_PUNCT = ',.。!！?？…—–,，。、；：'
        
        # 合并相邻的短chunk
        merged = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            stripped = chunk.strip()
            if not stripped:
                i += 1
                continue
            
            # 规则1：以标点开头 → 合并到上一个
            if stripped[0] in TRAILING_PUNCT:
                logger.debug(f"[PostProcess] 合并标点开头: {stripped[:30]}")
                if merged:
                    merged[-1] = merged[-1] + stripped
                    i += 1
                    continue
            
            # 规则2：以图表标记开头 → 合并到上一个
            if stripped.startswith(('表：', '图：', '表:', '图:')):
                logger.debug(f"[PostProcess] 合并图表开头: {stripped[:30]}")
                if merged:
                    merged[-1] = merged[-1] + '\n' + stripped
                    i += 1
                    continue
            
            # 规则3：以开括号开头（如（、"、【等）→ 合并到上一个
            if stripped[0] in OPEN_BRACKETS:
                logger.debug(f"[PostProcess] 合并括号开头: {stripped[:30]}")
                if merged:
                    merged[-1] = merged[-1] + stripped
                    i += 1
                    continue
            
            merged.append(chunk)
            i += 1
        
        logger.info(f"[PostProcess] 合并短尾后: {len(merged)} chunks")
        
        # 再次遍历：合并不完整的句子
        # 检查当前chunk的末尾和下一个chunk的开头是否匹配
        result = []
        i = 0
        while i < len(merged):
            chunk = merged[i]
            stripped = chunk.strip()
            
            if not stripped:
                i += 1
                continue
            
            # 如果当前chunk太短，尝试与下一个合并
            if len(stripped) < 30 and i + 1 < len(merged):
                next_chunk = merged[i + 1]
                next_stripped = next_chunk.strip()
                # 不要拆分表格、代码块
                if not next_stripped.startswith('|') and not next_stripped.startswith('```'):
                    merged_chunk = chunk + '\n' + next_chunk
                    logger.debug(f"[PostProcess] 合并短chunk ({len(stripped)}字): {stripped[:30]}")
                    result.append(merged_chunk)
                    i += 2
                    continue
            
            result.append(chunk)
            i += 1
        
        # 最终检查：处理后仍有问题的chunk
        still_bad = []
        for c in result:
            s = c.strip()
            if s and s[0] in TRAILING_PUNCT + OPEN_BRACKETS:
                still_bad.append((s[:50], s[-20:]))
        if still_bad:
            logger.warning(f"[PostProcess] 处理后仍有{len(still_bad)}个问题chunk")
        
        logger.info(f"[PostProcess] 最终输出: {len(result)} chunks")
        return result

    def _post_process_chunks_past(self, chunks: List[str]) -> List[str]:
        """
        后处理chunks：过滤噪声、合并短chunk

        处理规则：
        1. 合并孤立的图片链接到上下文
        2. 过滤孤立的过短数学公式（如 $@$）
        3. 合并过短的chunk到上一个chunk
        """
        if not chunks:
            return chunks

        # 正则模式
        image_pattern = re.compile(r'^!\[[^\]]*\]\([^)]+\)$')
        math_pattern = re.compile(r'^\$[\s\S]*?\$$')

        processed = []

        for chunk in chunks:
            stripped = chunk.strip()

            # 规则1：过滤孤立且过短的图片链接
            if image_pattern.match(stripped) and len(stripped) < 200:
                # 合并到上一个chunk
                if processed:
                    processed[-1] = processed[-1] + "\n" + stripped
                continue

            # 规则2：过滤孤立的过短数学公式（如 $@$）
            if math_pattern.match(stripped) and len(stripped) < 20:
                # 合并到上一个chunk
                if processed:
                    processed[-1] = processed[-1] + " " + stripped
                continue

            processed.append(chunk)

        # 规则3：将短标题合并到下一个chunk（标题本身没有分块价值）
        result = []
        i = 0
        while i < len(processed):
            chunk = processed[i]
            stripped = chunk.strip()

            # 检查是否是短标题（以#开头且长度<100）
            is_short_heading = stripped.startswith('#') and len(stripped) < 100

            if is_short_heading and i + 1 < len(processed):
                # 短标题和下一个chunk合并
                next_chunk = processed[i + 1]
                merged = chunk + "\n" + next_chunk
                result.append(merged)
                i += 2  # 跳过已合并的两个
                continue
            elif is_short_heading:
                # 标题是最后一个，说明前面已经合并过了，合并到上一个
                if result:
                    result[-1] = result[-1] + "\n" + chunk
                else:
                    result.append(chunk)
                i += 1
                continue

            # 如果当前chunk太短（非标题），合并到上一个
            if len(stripped) < 50 and result:
                if not stripped.startswith('|') and not stripped.startswith('```'):
                    result[-1] = result[-1] + "\n" + chunk
                    i += 1
                    continue

            result.append(chunk)
            i += 1

        return result

    def _build_splitting_units(self, text: str, protected_spans: List[ProtectedContentSpan], 
                              separators: List[str]) -> List[Tuple[str, int, int, bool]]:
        """
        构建分割单元
        
        Returns:
            List[Tuple[content, start_pos, end_pos, is_protected]]
        """
        units = []
        last_pos = 0
        
        # 按位置处理受保护的内容
        for span in protected_spans:
            # 添加受保护内容前的普通文本
            if span.start > last_pos:
                normal_text = text[last_pos:span.start]
                normal_units = self._split_normal_text(normal_text, separators)
                for unit_content, unit_start, unit_end in normal_units:
                    units.append((
                        unit_content, 
                        unit_start + last_pos, 
                        unit_end + last_pos, 
                        False
                    ))
            
            # 添加受保护的内容
            units.append((
                text[span.start:span.end],
                span.start,
                span.end,
                True
            ))
            
            last_pos = span.end
        
        # 添加剩余的文本
        if last_pos < len(text):
            remaining_text = text[last_pos:]
            normal_units = self._split_normal_text(remaining_text, separators)
            for unit_content, unit_start, unit_end in normal_units:
                units.append((
                    unit_content,
                    unit_start + last_pos,
                    unit_end + last_pos,
                    False
                ))
        
        return units

    def _split_normal_text(self, text: str, separators: List[str]) -> List[Tuple[str, int, int]]:
        """分割普通文本"""
        if not text.strip():
            return []
        
        # 构建分割正则表达式
        escaped_separators = [re.escape(sep) for sep in separators]
        pattern = '(' + '|'.join(escaped_separators) + ')'
        regex = re.compile(pattern)
        
        # 分割文本
        parts = regex.split(text)
        units = []
        current_pos = 0
        
        i = 0
        while i < len(parts):
            part = parts[i]
            if part:  # 非空部分
                units.append((part, current_pos, current_pos + len(part)))
                current_pos += len(part)
            
            # 如果下一个部分是分隔符，也要包含进去
            if i + 1 < len(parts) and parts[i + 1] in separators:
                sep = parts[i + 1]
                units.append((sep, current_pos, current_pos + len(sep)))
                current_pos += len(sep)
                i += 2
            else:
                i += 1
        
        return units

    def _merge_units_to_chunks(self, units: List[Tuple[str, int, int, bool]], 
                              max_chunk_size: int) -> List[str]:
        """将分割单元合并成chunks
        
        核心原则：受保护内容（表格、图片、代码块等）必须完整保留，
        即使超过 max_chunk_size 也不能被分割。
        """
        chunks = []
        current_chunk = ""
        current_size = 0
        
        i = 0
        while i < len(units):
            content, start_pos, end_pos, is_protected = units[i]
            
            content_size = len(content.encode('utf-8'))
            
            # 核心逻辑：受保护内容必须完整保留，不能分割
            if is_protected:
                # 受保护内容：先保存当前chunk，再单独成为一个chunk
                if current_chunk:
                    stripped = current_chunk.strip()
                    if stripped:
                        chunks.append(stripped)
                # 受保护内容单独作为一个chunk（即使超长）
                stripped = content.strip()
                if stripped:
                    chunks.append(stripped)
                current_chunk = ""
                current_size = 0
            elif current_chunk and current_size + content_size > max_chunk_size:
                # 普通内容，超出限制了，保存当前chunk，开始新的
                stripped = current_chunk.strip()
                if stripped:
                    chunks.append(stripped)
                current_chunk = content
                current_size = content_size
            else:
                # 普通内容，可以添加到当前chunk
                current_chunk += content
                current_size += content_size
            
            i += 1
        
        # 添加最后一个chunk
        stripped = current_chunk.strip()
        if stripped and stripped[0] not in ',.。!！?？':
            chunks.append(stripped)
        elif stripped and stripped[0] in ',.。!！?？':
            if chunks:
                chunks[-1] = chunks[-1] + stripped
            else:
                chunks.append(stripped)
        
        return chunks


def test_document_structure_analyzer():
    """测试文档结构分析器"""
    analyzer = DocumentStructureAnalyzer()
    
    # 测试文本
    test_text = """# 中国企业级AI应用行业研究报告

## 摘要

应用现状：随着"百模大战"逐渐落幕，行业竞争重心转变，企业级AI从技术探索期全面转向规模化应用期。

![](images/sample.jpg)

关键技术：
- 大语言模型能力跃升
- Agent成为核心载体
- 数据底座建设

$$E = mc^2$$

```python
def hello():
    print("Hello World")
```

## 发展趋势

未来展望：AI应用将深度介入企业流程，人机协作模式将发生转变。"""

    print("=== 测试文档结构分析 ===")
    
    # 分析结构
    elements, protected_spans = analyzer.analyze_structure(test_text)
    
    print(f"发现 {len(elements)} 个结构元素:")
    for i, elem in enumerate(elements):
        print(f"  {i+1}. [{elem.type}] {elem.content[:50]}...")
    
    print(f"\n发现 {len(protected_spans)} 个受保护内容:")
    for span in protected_spans:
        content_preview = test_text[span.start:span.end][:30]
        print(f"  [{span.type}] {content_preview}...")
    
    # 测试智能分割
    print("\n=== 测试智能分割 ===")
    chunks = analyzer.smart_split_text(test_text, max_chunk_size=200)
    print(f"分割成 {len(chunks)} 个chunk:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} ({len(chunk)}字符): {chunk[:100]}...")


if __name__ == "__main__":
    test_document_structure_analyzer()