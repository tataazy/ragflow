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
        self.separators = separators or ["\n\n", "。", "！", "？", ".", "!", "?"]
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
        """将图片分配到对应的文本chunks中
        
        使用基于内容重叠的分配策略，而非依赖位置索引。
        """
        self.logger.info(f"[SmartChunker] 开始图片分配 - 原始文本段落数: {len(original_texts)}, 图片数: {len(images)}, chunks数: {len(text_chunks)}")
        
        if not images or not text_chunks:
            self.logger.warning(f"[SmartChunker] 图片或chunks为空，返回全None数组")
            return [None] * len(text_chunks)
        
        # 统计图片分布
        total_images = len([img for img in images if img is not None])
        self.logger.info(f"[SmartChunker] 原始数据中有 {total_images} 张非None图片")
        
        # 方法1：使用语义重叠匹配（更可靠）
        image_chunks = self._semantic_image_distribution(original_texts, images, text_chunks)
        
        final_count = len([img for img in image_chunks if img is not None])
        self.logger.info(f"[SmartChunker] 语义匹配分配完成，分配了 {final_count} 张图片到chunks")
        
        if final_count < total_images * 0.5:
            # 如果语义匹配效果不好，尝试备用方案
            self.logger.warning(f"[SmartChunker] 语义匹配效果不佳({final_count}/{total_images})，尝试备用方案")
            fallback_chunks = self._sequential_image_distribution(original_texts, images, text_chunks)
            fallback_count = len([img for img in fallback_chunks if img is not None])
            if fallback_count > final_count:
                self.logger.info(f"[SmartChunker] 备用方案更好({fallback_count}/{total_images})，使用备用方案")
                return fallback_chunks
        
        return image_chunks

    def _semantic_image_distribution(self, original_texts: List[str], images: List, 
                                    text_chunks: List[str]) -> List:
        """基于内容重叠的图片分配
        
        对于每个chunk，找到所有与其内容重叠的原始段落，
        如果这些段落中有图片，则将图片分配给该chunk。
        """
        image_chunks = []
        
        for chunk_idx, chunk_text in enumerate(text_chunks):
            if not chunk_text:
                image_chunks.append(None)
                continue
            
            chunk_images = []
            
            # 策略：对每个原始段落，检查其内容是否出现在chunk中
            for seg_idx, (orig_text, image) in enumerate(zip(original_texts, images)):
                if image is None or not orig_text:
                    continue
                
                # 简化匹配：只检查较短的文本片段是否包含在chunk中
                # 或chunk是否包含较短的文本片段
                orig_len = len(orig_text)
                chunk_len = len(chunk_text)
                
                # 选择较小的文本作为搜索目标，提高匹配成功率
                if orig_len < chunk_len and orig_len > 5:
                    # 检查原始段落是否在chunk中
                    if orig_text in chunk_text:
                        chunk_images.append((seg_idx, image))
                        self.logger.debug(f"[SmartChunker] chunk {chunk_idx} ← 段落 {seg_idx} (精确匹配)")
                elif chunk_len < orig_len and chunk_len > 5:
                    # 检查chunk是否在原始段落中（不太可能，但试试无妨）
                    if chunk_text in orig_text:
                        chunk_images.append((seg_idx, image))
                        self.logger.debug(f"[SmartChunker] chunk {chunk_idx} ← 段落 {seg_idx} (反向匹配)")
            
            # 分配图片
            if chunk_images:
                # 选择第一个匹配的图片（简化处理）
                selected_image = chunk_images[0][1]
                image_chunks.append(selected_image)
                self.logger.debug(f"[SmartChunker] chunk {chunk_idx} 分配图片来自段落 {chunk_images[0][0]}")
            else:
                image_chunks.append(None)
        
        return image_chunks

    def _sequential_image_distribution(self, original_texts: List[str], images: List,
                                     text_chunks: List[str]) -> List:
        """顺序分配策略
        
        按顺序遍历sections，维护一个累积的文本缓冲区。
        当累积文本达到一个chunk大小时，创建一个新的chunk，
        并将之前所有包含图片的section的图片累积到该chunk。
        """
        image_chunks = []
        
        # 计算每个chunk的目标token数（估算）
        avg_chunk_size = sum(len(c) for c in text_chunks) / len(text_chunks) if text_chunks else 1000
        self.logger.info(f"[SmartChunker] 顺序分配策略 - 平均chunk大小: {avg_chunk_size:.0f} 字符")
        
        # 构建section列表（过滤空文本但保留索引映射）
        valid_sections = []
        for i, (text, image) in enumerate(zip(original_texts, images)):
            if text:  # 只保留非空文本
                valid_sections.append({
                    'original_idx': i,
                    'text': text,
                    'image': image,
                    'length': len(text)
                })
        
        # 按顺序分配图片到chunks
        section_idx = 0
        for chunk_idx, chunk_text in enumerate(text_chunks):
            if not chunk_text:
                image_chunks.append(None)
                continue
            
            chunk_len = len(chunk_text)
            
            # 找到所有与当前chunk对应的sections
            # 策略：从当前位置开始，收集足够覆盖chunk长度的sections
            accumulated_text = ""
            accumulated_images = []
            corresponding_sections = []
            
            while section_idx < len(valid_sections) and len(accumulated_text) < chunk_len * 1.2:
                section = valid_sections[section_idx]
                accumulated_text += "\n" + section['text']
                if section['image'] is not None:
                    accumulated_images.append(section['image'])
                corresponding_sections.append(section['original_idx'])
                section_idx += 1
            
            # 检查当前chunk是否在这个累积范围内
            if chunk_text in accumulated_text or accumulated_text in chunk_text:
                # 如果有图片，分配第一张
                if accumulated_images:
                    image_chunks.append(accumulated_images[0])
                    self.logger.debug(f"[SmartChunker] chunk {chunk_idx} 顺序分配来自sections {corresponding_sections[0]}-{corresponding_sections[-1]}")
                else:
                    image_chunks.append(None)
            else:
                # chunk不在累积范围内，说明有遗漏
                # 回退一些sections
                section_idx = max(0, section_idx - 2)
                image_chunks.append(None)
        
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
            # 不要使用 \n 作为分隔符，只用段落分隔和标点
            separators=config_dict.get("separators", ["\n\n", "。", "！", "？"]),
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
        chunk_size=chunk_token_num * 3,  # 估算：1 token ≈ 4 bytes
        chunk_overlap=int(chunk_token_num * overlapped_percent / 100 * 3),
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
    # target_chunks = max(10, min(200, len(chunks) * 2))  # 目标chunk数量范围
    
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
        chunk_size=chunk_token_num * 3,
        chunk_overlap=int(chunk_token_num * overlapped_percent / 100 * 3),
        separators=[d for d in delimiter] if isinstance(delimiter, str) else delimiter
    )
    
    chunker = SmartChunker(config)
    return chunker.split_with_images(texts, images, chunk_token_num * 3,
                                   int(chunk_token_num * overlapped_percent / 100 * 3))


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