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
from typing import List, Dict, Tuple, Optional, Union
import logging
from rag.utils.document_structure import DocumentStructureAnalyzer, ProtectedContentSpan


class SmartChunkConfig:
    """智能chunk配置"""
    def __init__(self, 
                 chunk_size: int = 512,
                 chunk_overlap: int = 128,
                 separators: List[str] = None,
                 preserve_elements: List[str] = None,
                 strategy: str = "structure_aware"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
        self.preserve_elements = preserve_elements or ["table", "image", "code_block", "math_block"]
        self.strategy = strategy


class SmartChunker:
    """智能文档chunk分割器"""
    
    def __init__(self, config: SmartChunkConfig = None):
        self.config = config or SmartChunkConfig()
        self.structure_analyzer = DocumentStructureAnalyzer()
        self.logger = logging.getLogger(__name__)

    def split_document(self, content: Union[str, List[Tuple[str, str]]], 
                      content_type: str = "markdown") -> List[str]:
        """
        智能分割文档
        
        Args:
            content: 文档内容，可以是字符串或(文本,位置)元组列表
            content_type: 内容类型 ("markdown", "text", "sections")
            
        Returns:
            List[str]: 分割后的chunk列表
        """
        if content_type == "markdown":
            return self._split_markdown(content)
        elif content_type == "sections":
            return self._split_sections(content)
        else:
            return self._split_plain_text(content)

    def _split_markdown(self, markdown_content: str) -> List[str]:
        """分割Markdown内容"""
        try:
            # 使用结构感知的方式分割
            chunks = self.structure_analyzer.smart_split_text(
                markdown_content,
                max_chunk_size=self.config.chunk_size,
                separators=self.config.separators
            )
            self.logger.info(f"Markdown内容分割成 {len(chunks)} 个chunks")
            return chunks
        except Exception as e:
            self.logger.warning(f"结构化分割失败，回退到简单分割: {e}")
            return self._fallback_split(markdown_content)

    def _split_sections(self, sections: List[Tuple[str, str]]) -> List[str]:
        """分割sections格式内容"""
        if not sections:
            return []
        
        # 将sections合并为文本
        combined_text = ""
        for text, pos in sections:
            if isinstance(text, str) and text.strip():
                combined_text += "\n" + text
        
        return self._split_markdown(combined_text.strip())

    def _split_plain_text(self, text: str) -> List[str]:
        """分割纯文本"""
        return self._fallback_split(text)

    def _fallback_split(self, text: str) -> List[str]:
        """回退分割方法 - 类似原有的naive_merge逻辑"""
        if not text or not text.strip():
            return []
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        # 按换行符分割
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            line_size = len(line.encode('utf-8'))
            
            # 如果当前chunk加上新行会超出限制
            if current_chunk and current_size + line_size > self.config.chunk_size:
                # 保存当前chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # 处理重叠
                if self.config.chunk_overlap > 0:
                    overlap_text = self._get_overlap_text(current_chunk, self.config.chunk_overlap)
                    current_chunk = overlap_text + "\n" + line
                else:
                    current_chunk = line
                current_size = len(current_chunk.encode('utf-8'))
            else:
                # 添加到当前chunk
                if current_chunk:
                    current_chunk += "\n" + line
                else:
                    current_chunk = line
                current_size += line_size + 1  # +1 for newline
        
        # 添加最后一个chunk
        if current_chunk and current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        self.logger.info(f"回退分割生成 {len(chunks)} 个chunks")
        return chunks

    def _get_overlap_text(self, text: str, overlap_chars: int) -> str:
        """获取重叠文本"""
        if not text or overlap_chars <= 0:
            return ""
        
        # 简单的重叠策略：取最后的字符
        text_bytes = text.encode('utf-8')
        if len(text_bytes) <= overlap_chars:
            return text
        
        # 找到合适的UTF-8边界
        overlap_start = len(text_bytes) - overlap_chars
        while overlap_start > 0 and (text_bytes[overlap_start] & 0xC0) == 0x80:
            overlap_start -= 1
        
        try:
            return text_bytes[overlap_start:].decode('utf-8')
        except UnicodeDecodeError:
            return text[-overlap_chars:]  # fallback

    def split_with_images(self, texts: List[str], images: List, 
                         chunk_size: int = None, chunk_overlap: int = None) -> Tuple[List[str], List]:
        """
        带图片的智能分割
        
        Args:
            texts: 文本列表
            images: 图片列表（与texts对应）
            chunk_size: chunk大小
            chunk_overlap: 重叠大小
            
        Returns:
            Tuple[List[str], List]: (文本chunks, 对应的图片chunks)
        """
        if chunk_size is None:
            chunk_size = self.config.chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.config.chunk_overlap
            
        if not texts or len(texts) != len(images):
            return [], []
        
        # 合并文本进行智能分割
        combined_text = "\n".join([t for t in texts if t])
        text_chunks = self._split_markdown(combined_text)
        
        # 为每个文本chunk分配对应的图片
        image_chunks = self._distribute_images_to_chunks(texts, images, text_chunks)
        
        return text_chunks, image_chunks

    def _distribute_images_to_chunks(self, original_texts: List[str], images: List, 
                                   text_chunks: List[str]) -> List:
        """将图片分配到对应的文本chunks中"""
        self.logger.info(f"[SmartChunker] 开始图片分配 - 原始文本段落数: {len(original_texts)}, 图片数: {len(images)}, chunks数: {len(text_chunks)}")
        
        if not images or not text_chunks:
            self.logger.warning(f"[SmartChunker] 图片或chunks为空，返回全None数组")
            return [None] * len(text_chunks)
        
        # 构建原始文本在合并文本中的位置映射
        combined_text = "\n".join([t for t in original_texts if t])
        self.logger.info(f"[SmartChunker] 合并文本长度: {len(combined_text)} 字符")
        
        positions = []
        current_pos = 0
        
        for orig_text in original_texts:
            if orig_text:
                start = combined_text.find(orig_text, current_pos)
                if start != -1:
                    end = start + len(orig_text)
                    positions.append((start, end, orig_text))
                    current_pos = end
                    self.logger.debug(f"[SmartChunker] 定位文本段落: 位置[{start}:{end}], 内容预览: {orig_text[:50]}")
                else:
                    positions.append((current_pos, current_pos, orig_text))
                    self.logger.warning(f"[SmartChunker] 无法定位文本段落: {orig_text[:50]}")
            else:
                positions.append((current_pos, current_pos, ""))
        
        image_chunks = []
        
        # 基于位置重叠分配图片到chunks
        for i, chunk_text in enumerate(text_chunks):
            chunk_start = combined_text.find(chunk_text)
            if chunk_start == -1:
                self.logger.warning(f"[SmartChunker] 无法定位chunk {i} 在合并文本中")
                image_chunks.append(None)
                continue
                
            chunk_end = chunk_start + len(chunk_text)
            self.logger.debug(f"[SmartChunker] 处理chunk {i}: 位置[{chunk_start}:{chunk_end}], 长度: {len(chunk_text)}")
            
            # 找到与当前chunk位置重叠的所有原始段落
            overlapping_images = []
            for j, (start, end, orig_text) in enumerate(positions):
                if images[j] is not None:
                    # 检查位置区间是否有重叠
                    if start < chunk_end and end > chunk_start:
                        overlapping_images.append((j, images[j]))
                        self.logger.debug(f"[SmartChunker] chunk {i} 与段落 {j} 重叠: 段落位置[{start}:{end}], chunk位置[{chunk_start}:{chunk_end}]")
            
            # 如果有重叠的图片，选择第一个或者合并
            if overlapping_images:
                selected_image = overlapping_images[0][1]  # 选择第一个重叠的图片
                self.logger.info(f"[SmartChunker] chunk {i} 分配图片: 来自段落 {overlapping_images[0][0]}, 图片类型: {type(selected_image).__name__}")
                image_chunks.append(selected_image)
            else:
                self.logger.debug(f"[SmartChunker] chunk {i} 无重叠图片")
                image_chunks.append(None)
        
        final_non_none = len([img for img in image_chunks if img is not None])
        self.logger.info(f"[SmartChunker] 图片分配完成 - 总chunks: {len(image_chunks)}, 有图片的chunks: {final_non_none}")
        
        return image_chunks

    def _concat_images(self, img1, img2):
        """合并两张图片（简化实现）"""
        # 这里应该实现实际的图片合并逻辑
        # 目前简单返回第一张图片
        return img1 if img1 is not None else img2


def create_smart_chunker(config_dict: Dict = None) -> SmartChunker:
    """创建智能chunker实例"""
    if config_dict:
        config = SmartChunkConfig(
            chunk_size=config_dict.get("chunk_size", 512),
            chunk_overlap=config_dict.get("chunk_overlap", 128),
            separators=config_dict.get("separators", ["\n\n", "\n", "。", "！", "？"]),
            preserve_elements=config_dict.get("preserve_elements", 
                                            ["table", "image", "code_block", "math_block"]),
            strategy=config_dict.get("strategy", "structure_aware")
        )
    else:
        config = SmartChunkConfig()
    
    return SmartChunker(config)


# 兼容原有接口的函数
def smart_merge(sections: Union[str, List], chunk_token_num=128, 
                delimiter="\n。；！？", overlapped_percent=0, **kwargs) -> List[str]:
    """
    智能合并函数，兼容原有naive_merge接口
    
    Args:
        sections: 文本段落列表
        chunk_token_num: chunk token数量限制
        delimiter: 分隔符
        overlapped_percent: 重叠百分比
        **kwargs: 其他配置参数
        
    Returns:
        List[str]: 合并后的chunks
    """
    # 创建配置
    config = SmartChunkConfig(
        chunk_size=chunk_token_num * 4,  # 估算：1 token ≈ 4 bytes
        chunk_overlap=int(chunk_token_num * overlapped_percent / 100 * 4),
        separators=[d for d in delimiter] if isinstance(delimiter, str) else delimiter
    )
    
    # 创建chunker
    chunker = SmartChunker(config)
    
    # 处理输入
    if isinstance(sections, str):
        content = sections
        content_type = "text"
    elif isinstance(sections, list) and sections and isinstance(sections[0], tuple):
        content = sections
        content_type = "sections"
    elif isinstance(sections, list):
        content = "\n".join([str(s) for s in sections if s])
        content_type = "text"
    else:
        content = str(sections)
        content_type = "text"
    
    # 执行分割
    chunks = chunker.split_document(content, content_type)
    
    # 如果chunks太少或太多，可以调整策略
    target_chunks = max(10, min(200, len(chunks) * 2))  # 目标chunk数量范围
    
    return chunks


def smart_merge_with_images(texts, images, chunk_token_num=128, 
                          delimiter="\n。；！？", overlapped_percent=0, **kwargs):
    """
    带图片的智能合并函数
    
    Args:
        texts: 文本列表
        images: 图片列表
        chunk_token_num: chunk token数量限制
        delimiter: 分隔符
        overlapped_percent: 重叠百分比
        **kwargs: 其他配置参数
        
    Returns:
        Tuple[List[str], List]: (文本chunks, 图片chunks)
    """
    config = SmartChunkConfig(
        chunk_size=chunk_token_num * 4,
        chunk_overlap=int(chunk_token_num * overlapped_percent / 100 * 4),
        separators=[d for d in delimiter] if isinstance(delimiter, str) else delimiter
    )
    
    chunker = SmartChunker(config)
    return chunker.split_with_images(texts, images, chunk_token_num * 4, 
                                   int(chunk_token_num * overlapped_percent / 100 * 4))


def test_smart_chunker():
    """测试智能chunker"""
    # 测试配置
    config = SmartChunkConfig(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？"],
        preserve_elements=["table", "image", "code_block"]
    )
    
    chunker = SmartChunker(config)
    
    # 测试文本
    test_text = """# 测试文档

这是一个测试段落，包含一些重要内容。

## 第一节

这里有一些要点：
- 要点一
- 要点二
- 要点三

![](images/test.jpg)

代码示例：
```python
def test():
    print("Hello")
```

## 第二节

更多内容在这里，这部分比较长，用来测试分割效果。我们需要确保分割是合理的，不会破坏语义完整性。"""

    print("=== 智能Chunk分割测试 ===")
    
    chunks = chunker.split_document(test_text, "markdown")
    
    print(f"原文长度: {len(test_text)} 字符")
    print(f"分割成 {len(chunks)} 个chunks:")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({len(chunk)}字符):")
        print("-" * 40)
        print(chunk)
        print("-" * 40)


if __name__ == "__main__":
    test_smart_chunker()