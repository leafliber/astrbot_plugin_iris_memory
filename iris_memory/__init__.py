"""Iris Memory 插件业务包。

本包内的模块一律使用包内相对导入；main.py 通过
``from .iris_memory.xxx import ...`` 引用本包，使所有子模块都注册在
AstrBot 的插件模块前缀（data.plugins.<插件目录名>.*）之下，
从而被 AstrBot 的插件重载/更新机制完整清理。
"""
