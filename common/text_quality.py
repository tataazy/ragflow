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
    检测文本是否为乱码（性能优化版）

    Args:
        text: 待检测文本
        threshold: 乱码阈值，超过此值认为是乱码

    Returns:
        bool: 是否为乱码
    """
    if not text or len(text) < 10:
        return False

    # 快速路径：纯中文文本或主要为中文的文本
    chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
    total_chars = len(text)
    chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
    
    # 如果中文字符占比超过60%，认为是正常中文文本（考虑到标点和英文术语）
    if chinese_ratio > 0.6 and total_chars > 10:  # 增加长度限制避免极短文本误判
        return False
    
    # 完全由中文和空白字符组成的情况
    if re.match(r'^[\u4e00-\u9fa5\s]+$', text) and len(text) > 5:
        return False

    # 检测CID占位符 - 更严格的检测
    cid_matches = re.findall(r'\(cid:\d+\)', text)
    if len(cid_matches) > 0:
        cid_ratio = sum(len(m) for m in cid_matches) / len(text)
        # 对于CID占位符采用零容忍策略：只要存在就判定为乱码
        logging.debug(f"Detected CID placeholders: count={len(cid_matches)}, ratio={cid_ratio:.2%}")
        return True

    # 检测伪英文乱码
    if _is_pseudo_english_gibberish(text):
        logging.debug(f"Detected pseudo-English gibberish: {text[:100]}...")
        return True

    # 单次遍历统计
    total = len(text)
    ascii_count = 0
    control_count = 0
    char_counter = Counter()

    for c in text:
        code = ord(c)
        if code < 128:
            ascii_count += 1
        if code < 32 and c not in '\t\n\r':
            control_count += 1
        char_counter[c] += 1

    # 快速失败：控制字符过多
    if control_count / total > 0.1:
        return True

    # 重复字符检查
    most_common = char_counter.most_common(1)[0][1]
    if most_common / total > 0.5:
        return True

    # 特殊字符检查（增强版）
    special_chars = re.findall(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?`~]{3,}', text)
    special_ratio = sum(len(s) for s in special_chars) / total if total > 0 else 0
    
    # 检测异常Unicode字符
    unusual_unicode_chars = [0x22, 0x15, 0x6a, 0x9, 0x3f, 0xed, 0x77, 0x38, 0x201c, 0x201d,
                            0x2018, 0x2019, 0x2026, 0x2014, 0x2013, 0x141, 0x144, 0x14c, 0x157]
    unusual_unicode_count = sum(1 for c in text if ord(c) in unusual_unicode_chars)
    unusual_unicode_ratio = unusual_unicode_count / total if total > 0 else 0
    
    # 检测混合乱码模式（如示例中的 DÝ&/gE⁄,ž/›3 这种模式）
    # 排除正常的技术术语和专业符号
    tech_terms = r'(API|SDK|URL|DNA|RNA|HTTP|JSON|XML|SQL|CSS|HTML|PDF|CID|ID|NO|OK|US|UK|EU|USA|UK|CH|JP|KR|SG|AU|CA|DE|FR|IT|ES|NL|SE|NO|DK|FI|PL|CZ|HU|RO|BG|HR|SI|SK|EE|LV|LT|MT|CY|LU|BE|IE|PT|GR|AT|CH|LI|IS|AL|MK|ME|RS|BA|HR|SI|SK|EE|LV|LT)'
    
    # 更精确的混合乱码检测，排除合法技术术语
    mixed_garbage_pattern = re.search(r'(?<!\b)(?!' + tech_terms + r'\b)[A-Z][a-z]*[&/][A-Z][a-z]*[^a-zA-Z0-9\u4e00-\u9fa5\s]{2,}', text)
    has_mixed_garbage = bool(mixed_garbage_pattern)
    
    if unusual_unicode_ratio > 0.15:
        logging.debug(f"Detected unusual Unicode characters: {unusual_unicode_ratio:.2%}")
        return True
    
    # 检测混合乱码模式
    if has_mixed_garbage:
        logging.debug(f"Detected mixed garbage pattern: {mixed_garbage_pattern.group()}")
        return True

    # 综合评分（排除中文干扰）
    chinese_count = len(re.findall(r'[\u4e00-\u9fa5]', text))
    non_chinese_total = total - chinese_count

    if non_chinese_total == 0:
        return False

    non_ascii_in_non_chinese = sum(1 for c in text if ord(c) >= 128 and not '\u4e00' <= c <= '\u9fa5')
    non_ascii_ratio = non_ascii_in_non_chinese / non_chinese_total

    # 对于技术文档适当放宽标准
    tech_indicators = len(re.findall(r'\b(API|SDK|URL|HTTP|JSON|XML|SQL|CSS|HTML|PDF)\b', text, re.IGNORECASE))
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