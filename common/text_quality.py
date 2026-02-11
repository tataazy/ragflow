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

import re
import logging
from collections import Counter


def _is_pseudo_english_gibberish(text):
    """
    检测"伪英文乱码" - 看起来像英文单词但实际是随机字符组合
    例如: "6SDFH;LCE SI 3 URMAESHDQH" 这种无意义的字符序列
    
    Returns:
        bool: 是否为伪英文乱码
    """
    if not text or len(text) < 10:
        return False
    
    # 只检查包含字母的文本
    letters = re.findall(r'[a-zA-Z]+', text)
    if not letters:
        return False
    
    # 合并所有单词进行检查
    all_letters = ''.join(letters).upper()
    if len(all_letters) < 8:
        return False
    
    # 1. 检测辅音群比例（英文中连续的辅音通常不超过3个）
    consonant_clusters = re.findall(r'[BCDFGHJKLMNPQRSTVWXZ]{4,}', all_letters)
    if consonant_clusters:
        # 有超过4个连续辅音，可能是乱码
        return True
    
    # 2. 检测元音比例（英文中元音占比约40%）
    vowels = re.findall(r'[AEIOU]', all_letters)
    vowel_ratio = len(vowels) / len(all_letters) if all_letters else 0
    
    # 元音比例异常低或高，可能是乱码
    if vowel_ratio < 0.15 or vowel_ratio > 0.6:
        return True
    
    # 3. 检测常见双字母组合（bigrams）比例
    # 英文中某些字母组合很常见（如TH, HE, IN, ER等）
    common_bigrams = {'TH', 'HE', 'IN', 'ER', 'AN', 'RE', 'ON', 'AT', 'EN', 'ND', 
                      'TI', 'ES', 'OR', 'TE', 'OF', 'ED', 'IS', 'IT', 'AL', 'AR',
                      'ST', 'TO', 'NT', 'EA', 'NG', 'AS', 'OU', 'SE', 'HA', 'ND'}
    
    bigram_count = 0
    for i in range(len(all_letters) - 1):
        if all_letters[i:i+2] in common_bigrams:
            bigram_count += 1
    
    bigram_ratio = bigram_count / (len(all_letters) - 1) if len(all_letters) > 1 else 0
    
    # 常见字母组合比例过低，可能是乱码
    # 英文中常见bigram比例通常在0.3-0.5之间
    if bigram_ratio < 0.15:
        return True
    
    # 4. 检测连续数字和字母混合模式
    # 乱码中常出现数字和字母不规则混合
    mixed_pattern = re.findall(r'[0-9]+[A-Z]+[0-9]+', all_letters)
    if len(mixed_pattern) > 2:
        return True
    
    return False


def is_gibberish(text, threshold=0.3):
    """
    检测文本是否为乱码
    
    Args:
        text: 待检测文本
        threshold: 乱码阈值，超过此值认为是乱码
        
    Returns:
        bool: 是否为乱码
    """
    if not text or not text.strip():
        return False
    
    # 首先检测"伪英文乱码"（看起来像英文但实际是随机字符）
    if _is_pseudo_english_gibberish(text):
        logging.debug(f"Detected pseudo-English gibberish: {text[:100]}...")
        return True
    
    # 1. 检测非ASCII字符比例（针对英文文档）
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    total_chars = len(text)
    non_ascii_ratio = (total_chars - ascii_chars) / total_chars if total_chars > 0 else 0
    
    # 2. 检测控制字符比例
    control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\t\n\r')
    control_ratio = control_chars / total_chars if total_chars > 0 else 0
    
    # 3. 检测重复字符比例
    if total_chars > 0:
        char_counts = Counter(text)
        most_common_char, count = char_counts.most_common(1)[0]
        repeat_ratio = count / total_chars
    else:
        repeat_ratio = 0
    
    # 4. 检测无意义字符组合（例如连续的特殊字符）
    special_chars = re.findall(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?`~]{3,}', text)
    special_ratio = sum(len(s) for s in special_chars) / total_chars if total_chars > 0 else 0
    
    # 5. 综合评分
    gibberish_score = max(non_ascii_ratio, control_ratio, repeat_ratio, special_ratio)
    
    logging.debug(f"Text quality analysis: non_ascii={non_ascii_ratio:.2f}, control={control_ratio:.2f}, repeat={repeat_ratio:.2f}, special={special_ratio:.2f}, score={gibberish_score:.2f}")
    
    return gibberish_score > threshold


def contains_gibberish(text, threshold=0.5):
    """
    检测文本是否包含大量乱码内容
    
    Args:
        text: 待检测文本
        threshold: 乱码内容占比阈值
        
    Returns:
        bool: 是否包含大量乱码
    """
    if not text or not text.strip():
        return False
    
    # 分割文本为句子
    sentences = re.split(r'[.!?。！？；;\n]+', text)
    
    # 计算乱码句子比例
    gibberish_count = 0
    total_sentences = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            total_sentences += 1
            if is_gibberish(sentence):
                gibberish_count += 1
    
    if total_sentences == 0:
        return False
    
    gibberish_ratio = gibberish_count / total_sentences
    logging.debug(f"Gibberish sentence ratio: {gibberish_ratio:.2f}")
    
    return gibberish_ratio > threshold


def filter_gibberish(text, min_length=5):
    """
    过滤文本中的乱码内容
    
    Args:
        text: 待过滤文本
        min_length: 最小保留文本长度
        
    Returns:
        str: 过滤后的文本
    """
    if not text or not text.strip():
        return ""
    
    # 分割文本为句子
    sentences = re.split(r'([.!?。！？；;\n]+)', text)
    
    # 过滤乱码句子
    filtered_sentences = []
    for i in range(0, len(sentences), 2):
        sentence = sentences[i].strip()
        punctuation = sentences[i+1] if i+1 < len(sentences) else ""
        
        if sentence and not is_gibberish(sentence) and len(sentence) >= min_length:
            filtered_sentences.append(sentence + punctuation)
    
    filtered_text = "".join(filtered_sentences)
    logging.debug(f"Filtered gibberish: original length={len(text)}, filtered length={len(filtered_text)}")
    
    return filtered_text


def text_quality_score(text):
    """
    计算文本质量评分
    
    Args:
        text: 待评分文本
        
    Returns:
        float: 质量评分，0-1之间，越高越好
    """
    if not text or not text.strip():
        return 0.0
    
    text = text.strip()
    total_chars = len(text)
    
    # 1. 有效字符比例（排除控制字符和无意义字符）
    pattern = r'[a-zA-Z0-9\u4e00-\u9fa5\s，。！？；："（）【】《》,.!?:;()\[\]{}<>‘’'']'
    valid_chars = re.findall(pattern, text)
    valid_ratio = len(valid_chars) / total_chars if total_chars > 0 else 0
    
    # 2. 句子完整性（检测句子结束符）
    sentences = re.split(r'[.!?。！？]+', text)
    complete_sentences = [s for s in sentences if s.strip() and len(s.strip()) > 2]
    complete_ratio = len(complete_sentences) / len(sentences) if sentences else 0
    
    # 3. 词汇多样性
    words = re.findall(r'[a-zA-Z]+|\u4e00-\u9fa5+', text)
    unique_words = set(words)
    diversity_score = len(unique_words) / len(words) if words else 0
    
    # 4. 综合评分
    quality_score = (valid_ratio * 0.5 + complete_ratio * 0.3 + diversity_score * 0.2)
    
    logging.debug(f"Text quality score: valid={valid_ratio:.2f}, complete={complete_ratio:.2f}, diversity={diversity_score:.2f}, total={quality_score:.2f}")
    
    return quality_score