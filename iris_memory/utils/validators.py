"""
validators.py - 校验工具函数

从 sensitivity_detector 提取的独立校验逻辑，可被其他模块复用。
"""

from __future__ import annotations


def validate_china_id(digits: str) -> bool:
    """验证中国身份证号校验位（GB 11643-1999）

    Args:
        digits: 18位身份证号字符串

    Returns:
        校验位是否合法
    """
    if len(digits) != 18:
        return False
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    check_chars = '10X98765432'
    try:
        total = sum(int(digits[i]) * weights[i] for i in range(17))
        return check_chars[total % 11].upper() == digits[17].upper()
    except (ValueError, IndexError):
        return False


def validate_bank_card(digits: str) -> bool:
    """Luhn 算法验证银行卡号

    Args:
        digits: 16-19位纯数字字符串

    Returns:
        Luhn 校验是否通过
    """
    if not digits.isdigit() or len(digits) < 16 or len(digits) > 19:
        return False
    total = 0
    reverse_digits = digits[::-1]
    for i, ch in enumerate(reverse_digits):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0
