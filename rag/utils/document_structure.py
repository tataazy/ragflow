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
            'table_header': re.compile(r'(?:\|[^|\n]*)+\|[\\r\\n]+\s*(?:\|\\s*:?-{3,}:?\\s*)+\|'),
            'table_row': re.compile(r'(?:\|[^|\n]*)+\|[\\r\\n]+'),
            'code_block': re.compile(r'```(?:\\w+)?[\\r\\n].*?```', re.DOTALL),
            'inline_code': re.compile(r'`[^`]+`'),
            'math_inline': re.compile(r'\$[^$]+\$(?!\$)')
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
        智能分割文本，保护重要内容不被分割
        
        Args:
            text: 要分割的文本
            max_chunk_size: 最大chunk大小
            separators: 分割符列表，按优先级排序
            
        Returns:
            List[str]: 分割后的文本块列表
        """
        if separators is None:
            separators = ["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
        
        # 找出受保护的内容
        protected_spans = self._find_protected_spans(text)
        
        # 构建分割单元
        units = self._build_splitting_units(text, protected_spans, separators)
        
        # 合并单元成chunks
        chunks = self._merge_units_to_chunks(units, max_chunk_size)
        
        return chunks

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
        """将分割单元合并成chunks"""
        chunks = []
        current_chunk = ""
        current_size = 0
        
        i = 0
        while i < len(units):
            content, start_pos, end_pos, is_protected = units[i]
            
            content_size = len(content.encode('utf-8'))
            
            # 如果添加当前单元会超出大小限制
            if current_chunk and current_size + content_size > max_chunk_size:
                # 如果当前单元是受保护的且很大，需要特殊处理
                if is_protected and content_size > max_chunk_size:
                    # 受保护的大内容单独成为一个chunk
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    chunks.append(content.strip())
                    current_chunk = ""
                    current_size = 0
                else:
                    # 正常情况，先保存当前chunk，再开始新的
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = content
                    current_size = content_size
            else:
                # 可以添加到当前chunk
                current_chunk += content
                current_size += content_size
            
            i += 1
        
        # 添加最后一个chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
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