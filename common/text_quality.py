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


# 预编译正则表达式以提高性能
VOWELS_PATTERN = re.compile(r'[AEIOU]')
CONSONANT_CLUSTERS_PATTERN = re.compile(r'[BCDFGHJKLMNPQRSTVWXZ]{4,}')
MIXED_DIGIT_LETTER_PATTERN = re.compile(r'[0-9]+[A-Z]+[0-9]+')

# 常见双字母组合集合
COMMON_BIGRAMS = {'TH', 'HE', 'IN', 'ER', 'AN', 'RE', 'ON', 'AT', 'EN', 'ND', 
                  'TI', 'ES', 'OR', 'TE', 'OF', 'ED', 'IS', 'IT', 'AL', 'AR',
                  'ST', 'TO', 'NT', 'EA', 'NG', 'AS', 'OU', 'SE', 'HA', 'ND'}



def _is_pseudo_english_gibberish(text):
    """
    检测"伪英文乱码" - 看起来像英文单词但实际是随机字符组合
    例如: "6SDFH;LCE SI 3 URMAESHDQH" 这种无意义的字符序列
    
    Returns:
        bool: 是否为伪英文乱码
    """
    if not text or len(text) < 10:
        return False
    
    # 快速检查：是否包含足够的字母
    has_letters = any(c.isalpha() for c in text)
    if not has_letters:
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
    if CONSONANT_CLUSTERS_PATTERN.search(all_letters):
        # 有超过4个连续辅音，可能是乱码
        return True
    
    # 2. 检测元音比例（英文中元音占比约40%）
    vowel_count = len(VOWELS_PATTERN.findall(all_letters))
    vowel_ratio = vowel_count / len(all_letters) if all_letters else 0
    
    # 元音比例异常低或高，可能是乱码
    if vowel_ratio < 0.15 or vowel_ratio > 0.6:
        return True
    
    # 3. 检测常见双字母组合（bigrams）比例
    bigram_count = 0
    total_bigrams = len(all_letters) - 1
    if total_bigrams > 0:
        for i in range(total_bigrams):
            if all_letters[i:i+2] in COMMON_BIGRAMS:
                bigram_count += 1
        
        bigram_ratio = bigram_count / total_bigrams
        
        # 常见字母组合比例过低，可能是乱码
        if bigram_ratio < 0.15:
            return True
    
    # 4. 检测连续数字和字母混合模式
    mixed_pattern = MIXED_DIGIT_LETTER_PATTERN.findall(all_letters)
    if len(mixed_pattern) > 2:
        return True
    
    return False


# 预编译正则表达式以提高性能
CID_PLACEHOLDER_PATTERN = re.compile(r'\(cid:\d+\)')
REPLACEMENT_CHAR_PATTERN = re.compile(r'[\ufffd\ue000]')
CHINESE_CHAR_PATTERN = re.compile(r'[\u4e00-\u9fa5]')
SPECIAL_CHARS_PATTERN = re.compile(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?`~]{3,}')
TECH_TERMS_PATTERN = re.compile(r'\b(API|SDK|URL|HTTP|JSON|XML|SQL|CSS|HTML|PDF)\b', re.IGNORECASE)

# 异常Unicode字符集合，使用集合提高查找速度
UNUSUAL_UNICODE_CHARS = {0x22, 0x15, 0x6a, 0x9, 0x3f, 0xed, 0x77, 0x38, 0x201c, 0x201d,
                         0x2018, 0x2019, 0x2026, 0x2014, 0x2013, 0x141, 0x144, 0x14c, 0x157}

# 技术术语集合，用于快速查找
TECH_TERMS_SET = {'API', 'SDK', 'URL', 'DNA', 'RNA', 'HTTP', 'JSON', 'XML', 'SQL', 'CSS', 'HTML', 'PDF', 'CID', 'ID', 'NO', 'OK', 'US', 'UK', 'EU', 'USA', 'CH', 'JP', 'KR', 'SG', 'AU', 'CA', 'DE', 'FR', 'IT', 'ES', 'NL', 'SE', 'NO', 'DK', 'FI', 'PL', 'CZ', 'HU', 'RO', 'BG', 'HR', 'SI', 'SK', 'EE', 'LV', 'LT', 'MT', 'CY', 'LU', 'BE', 'IE', 'PT', 'GR', 'AT', 'CH', 'LI', 'IS', 'AL', 'MK', 'ME', 'RS', 'BA'}

# 混合乱码模式，使用更简单的模式提高性能
MIXED_GARBAGE_PATTERN = re.compile(r'[A-Z][a-z]*[&/][A-Z][a-z]*[^a-zA-Z0-9\u4e00-\u9fa5\s]{2,}')


def is_gibberish(text, threshold=0.3):
    """
    检测文本是否为乱码（优化版本，提高性能）

    Args:
        text: 待检测文本
        threshold: 乱码阈值，超过此值认为是乱码

    Returns:
        bool: 是否为乱码
    """
    if not text or len(text) < 10:
        return False

    # 快速路径1：检查替换字符（性能最优）
    if REPLACEMENT_CHAR_PATTERN.search(text):
        return True

    # 快速路径2：检查 CID 占位符
    if '(cid:' in text:
        cid_count = text.count('(cid:')
        if cid_count > 2 or cid_count / len(text) > 0.01:
            return True

    # 快速路径3：简单长度检查
    if len(text) > 1000:
        # 长文本只采样检查
        text = text[:500] + text[-500:]

    # 快速路径4：纯中文文本直接通过
    chinese_chars = len(CHINESE_CHAR_PATTERN.findall(text))
    if chinese_chars > len(text) * 0.3:
        return False

    # 单次遍历统计多个指标，减少循环次数
    total = len(text)
    ascii_count = 0
    control_count = 0
    chinese_count = 0
    unusual_unicode_count = 0
    non_ascii_in_non_chinese = 0
    char_counter = Counter()

    for c in text:
        code = ord(c)
        if code < 128:
            ascii_count += 1
        if code < 32 and c not in '\t\n\r':
            control_count += 1
        if '\u4e00' <= c <= '\u9fa5':
            chinese_count += 1
        if code in UNUSUAL_UNICODE_CHARS:
            unusual_unicode_count += 1
        if code >= 128 and not '\u4e00' <= c <= '\u9fa5':
            non_ascii_in_non_chinese += 1
        char_counter[c] += 1

    # 快速失败：控制字符过多
    if control_count / total > 0.1:
        return True

    # 重复字符检查
    most_common = char_counter.most_common(1)[0][1]
    if most_common / total > 0.5:
        return True

    # 特殊字符检查（增强版）
    special_chars = SPECIAL_CHARS_PATTERN.findall(text)
    special_ratio = sum(len(s) for s in special_chars) / total if total > 0 else 0
    
    unusual_unicode_ratio = unusual_unicode_count / total if total > 0 else 0
    
    # 快速检查异常Unicode字符比例
    if unusual_unicode_ratio > 0.15:
        logging.debug(f"Detected unusual Unicode characters: {unusual_unicode_ratio:.2%}")
        return True
    
    # 检测混合乱码模式（简化版本，提高性能）
    mixed_garbage_match = MIXED_GARBAGE_PATTERN.search(text)
    if mixed_garbage_match:
        # 检查是否是合法技术术语
        match_text = mixed_garbage_match.group()
        is_tech_term = any(term in match_text.upper() for term in TECH_TERMS_SET)
        if not is_tech_term:
            logging.debug(f"Detected mixed garbage pattern: {match_text}")
            return True

    # 检测伪英文乱码（移到后面，因为计算成本较高）
    if _is_pseudo_english_gibberish(text):
        logging.debug(f"Detected pseudo-English gibberish: {text[:100]}...")
        return True

    # 综合评分（排除中文干扰）
    non_chinese_total = total - chinese_count

    if non_chinese_total == 0:
        return False

    non_ascii_ratio = non_ascii_in_non_chinese / non_chinese_total

    # 对于技术文档适当放宽标准
    tech_indicators = len(TECH_TERMS_PATTERN.findall(text))
    if tech_indicators > 0 and total > 50:  # 只对较长的技术文档放宽标准
        # 有技术术语时，稍微提高阈值
        adjusted_threshold = threshold * 1.5
        gibberish_score = max(non_ascii_ratio, control_count / total, special_ratio, unusual_unicode_ratio)
        return gibberish_score > adjusted_threshold
    
    gibberish_score = max(non_ascii_ratio, control_count / total, special_ratio, unusual_unicode_ratio)

    logging.debug(f"Text quality analysis: non_ascii={non_ascii_ratio:.2f}, control={control_count/total:.2f}, special={special_ratio:.2f}, unusual_unicode={unusual_unicode_ratio:.2f}, score={gibberish_score:.2f}")

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


# 预编译句子分割正则表达式
SENTENCE_SPLIT_PATTERN = re.compile(r'([.!?。！？；;\n]+)')


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
    sentences = SENTENCE_SPLIT_PATTERN.split(text)
    
    # 过滤乱码句子
    filtered_sentences = []
    
    # 处理句子和标点符号对
    for i in range(0, len(sentences), 2):
        sentence = sentences[i].strip()
        punctuation = sentences[i+1] if i+1 < len(sentences) else ""
        
        # 跳过空句子
        if not sentence:
            continue
            
        # 检查长度阈值（快速检查，避免不必要的乱码检测）
        if len(sentence) < min_length:
            continue
            
        # 检查是否为乱码
        if not is_gibberish(sentence):
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