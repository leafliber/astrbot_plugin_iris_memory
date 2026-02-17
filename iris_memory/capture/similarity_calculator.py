"""
相似度计算模块

提供多种文本相似度计算算法，用于记忆去重和冲突检测。
"""

import re
from difflib import SequenceMatcher
from typing import Set


def sanitize_for_log(text: str, max_length: int = 50) -> str:
    """对文本进行脱敏处理后用于日志记录
    
    Args:
        text: 原始文本
        max_length: 最大长度
        
    Returns:
        str: 脱敏后的文本
    """
    if not text:
        return "[empty]"
    
    # 敏感模式替换
    sanitized = text
    
    # 手机号（11位数字）
    sanitized = re.sub(r'1[3-9]\d{9}', '[PHONE]', sanitized)
    # 身份证号
    sanitized = re.sub(r'\d{17}[\dXx]', '[ID_CARD]', sanitized)
    # 银行卡号（16-19位数字）
    sanitized = re.sub(r'\d{16,19}', '[BANK_CARD]', sanitized)
    # 密码相关
    sanitized = re.sub(r'密码[:：是]\S+', '密码:[MASKED]', sanitized)
    sanitized = re.sub(r'password[:：]\S+', 'password:[MASKED]', sanitized, flags=re.IGNORECASE)
    # 邮箱
    sanitized = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}', '[EMAIL]', sanitized)
    
    # 截断
    if len(sanitized) > max_length:
        return sanitized[:max_length] + "..."
    return sanitized


class SimilarityCalculator:
    """文本相似度计算器
    
    提供多种相似度计算算法：
    - 快速字符级相似度（Jaccard）
    - 精确多算法融合相似度（N-gram + Sequence + LCS）
    - 内容相似度（N-gram）
    - 共同主题检测
    """
    
    # 停用词集合
    STOPWORDS = {
        '的', '了', '在', '是', '我', '你', '他', '她', '它', '我们', '你们',
        '他们', '这', '那', '这些', '那些', '和', '与', '或', '就', '都', '而',
        '及', '与', '或', '但是', '然而', 'the', 'a', 'an', 'is', 'are', 'was',
        'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
        'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 'for', 'on',
        'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'under', 'again', 'further', 'then'
    }
    
    def calculate_quick_similarity(self, text1: str, text2: str) -> float:
        """快速相似度计算（用于预筛选）

        使用字符集合和哈希签名进行快速比较，时间复杂度 O(n)。

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            float: 相似度（0-1）
        """
        # 方法1: 字符集合Jaccard相似度
        set1 = set(text1.lower())
        set2 = set(text2.lower())

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        char_sim = intersection / union

        # 方法2: 词集合Jaccard相似度（更精确但稍慢）
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if words1 and words2:
            word_intersection = len(words1 & words2)
            word_union = len(words1 | words2)
            word_sim = word_intersection / word_union if word_union > 0 else 0.0
        else:
            word_sim = 0.0

        # 综合得分：字符相似度40% + 词相似度60%
        return 0.4 * char_sim + 0.6 * word_sim

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算精确文本相似度 - 多算法融合版

        使用三种互补的相似度算法，综合评估文本语义相似度：

        算法1 - N-gram相似度（权重40%）：
            使用2-gram捕捉局部字符模式
            对中文尤其有效，能识别相似词组和短语
            计算：|intersection(gram1, gram2)| / |union(gram1, gram2)|

        算法2 - 序列相似度（权重40%）：
            使用Python difflib.SequenceMatcher
            基于最长公共子序列（LCS）算法
            能识别整体结构相似性，对词序变化敏感

        算法3 - 最长公共子串（权重20%）：
            使用动态规划计算最长公共子串长度
            空间优化：滚动数组将O(M*N)降至O(N)
            识别连续匹配的片段

        综合公式：similarity = 0.4*ngram + 0.4*sequence + 0.2*lcs

        复杂度：
            时间：O(N*M)，N和M为文本长度
            空间：O(N)，使用滚动数组优化

        Args:
            text1: 待比较的文本1
            text2: 待比较的文本2

        Returns:
            float: 综合相似度得分（0-1）
        """
        text1_lower = text1.lower()
        text2_lower = text2.lower()

        # 方法1: N-gram相似度（捕捉局部模式）
        ngrams1 = self._get_ngrams(text1_lower, 2)
        ngrams2 = self._get_ngrams(text2_lower, 2)

        if ngrams1 and ngrams2:
            ngram_intersection = len(ngrams1 & ngrams2)
            ngram_union = len(ngrams1 | ngrams2)
            ngram_sim = ngram_intersection / ngram_union if ngram_union > 0 else 0.0
        else:
            ngram_sim = 0.0

        # 方法2: 序列相似度（使用difflib）
        seq_sim = SequenceMatcher(None, text1_lower, text2_lower).ratio()

        # 方法3: 公共子串比例
        max_len = max(len(text1_lower), len(text2_lower))
        if max_len > 0:
            lcs_len = self._longest_common_substring_length(text1_lower, text2_lower)
            lcs_sim = lcs_len / max_len
        else:
            lcs_sim = 0.0

        # 综合得分
        return 0.4 * ngram_sim + 0.4 * seq_sim + 0.2 * lcs_sim

    def calculate_content_similarity(self, text1: str, text2: str) -> float:
        """计算内容相似度（基于字符和词组）

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            float: 相似度 (0-1)
        """
        # 使用N-gram计算相似度
        ngrams1 = self._get_ngrams(text1.lower(), 2)
        ngrams2 = self._get_ngrams(text2.lower(), 2)

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        if union == 0:
            return 0.0

        return intersection / union

    def have_common_subject(self, text1: str, text2: str) -> bool:
        """检查两个文本是否有相同的主题/对象

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            bool: 是否有共同主题
        """
        # 提取关键词（长度大于1的词）
        words1 = set(w for w in re.findall(r'\w+', text1) if len(w) > 1 and w not in self.STOPWORDS)
        words2 = set(w for w in re.findall(r'\w+', text2) if len(w) > 1 and w not in self.STOPWORDS)

        if not words1 or not words2:
            return False

        # 如果有超过2个共同词，认为有共同主题
        common_words = words1 & words2
        return len(common_words) >= 2

    def _get_ngrams(self, text: str, n: int = 2) -> Set[str]:
        """获取文本的N-gram集合
        
        Args:
            text: 文本
            n: gram大小
            
        Returns:
            Set[str]: N-gram集合
        """
        return set(text[i:i+n] for i in range(len(text) - n + 1))

    def _longest_common_substring_length(self, s1: str, s2: str) -> int:
        """计算最长公共子串长度（动态规划）

        使用滚动数组优化空间复杂度从O(M*N)降至O(N)。

        Args:
            s1: 字符串1
            s2: 字符串2

        Returns:
            int: 最长公共子串长度
        """
        if not s1 or not s2:
            return 0

        m, n = len(s1), len(s2)
        # 使用滚动数组优化空间复杂度
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        max_length = 0

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    curr[j] = prev[j-1] + 1
                    max_length = max(max_length, curr[j])
                else:
                    curr[j] = 0
            prev, curr = curr, prev

        return max_length
