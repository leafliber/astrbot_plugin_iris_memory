#!/usr/bin/env python3
"""
Docker环境自动化测试脚本
用于验证Iris Memory插件在Docker环境中的功能
"""
import asyncio
import subprocess
import time
import requests
from typing import Dict, List

# 测试配置
WEBUI_URL = "http://localhost:6185"
TEST_USER_ID = "docker_test_user"
TEST_MESSAGES = [
    "我是Docker测试用户",
    "我喜欢编程和AI",
    "我觉得这个插件很棒",
    "我的工作是软件开发"
]

class DockerTester:
    """Docker环境测试器"""

    def __init__(self):
        self.webui_url = WEBUI_URL
        self.test_results = []

    def log(self, message: str, level: str = "INFO"):
        """记录日志"""
        prefix = {
            "INFO": "✓",
            "WARN": "⚠",
            "ERROR": "✗",
            "SUCCESS": "🎉"
        }.get(level, "•")
        print(f"{prefix} {message}")

    def check_docker_status(self) -> bool:
        """检查Docker状态"""
        try:
            result = subprocess.run(
                ["docker-compose", "ps"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return "astrbot-iris-memory" in result.stdout
        except Exception as e:
            self.log(f"检查Docker状态失败: {e}", "ERROR")
            return False

    def check_webui(self) -> bool:
        """检查WebUI是否可访问"""
        try:
            response = requests.get(self.webui_url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.log(f"WebUI不可访问: {e}", "ERROR")
            return False

    def get_container_logs(self, tail: int = 20) -> str:
        """获取容器日志"""
        try:
            result = subprocess.run(
                ["docker-compose", "logs", "--tail", str(tail), "astrbot"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout
        except Exception as e:
            return f"获取日志失败: {e}"

    def test_plugin_loaded(self) -> bool:
        """测试插件是否加载"""
        logs = self.get_container_logs(50)
        return "iris_memory" in logs.lower() and "plugin" in logs.lower()

    def run_test(self) -> Dict[str, bool]:
        """运行所有测试"""
        self.log("开始Docker环境测试", "INFO")
        self.log("=" * 50)

        results = {
            "docker_status": False,
            "webui_accessible": False,
            "plugin_loaded": False
        }

        # 测试1: Docker状态
        self.log("测试1: 检查Docker容器状态")
        results["docker_status"] = self.check_docker_status()
        if results["docker_status"]:
            self.log("Docker容器运行正常", "SUCCESS")
        else:
            self.log("Docker容器未运行", "ERROR")

        # 测试2: WebUI可访问性
        self.log("\n测试2: 检查WebUI可访问性")
        results["webui_accessible"] = self.check_webui()
        if results["webui_accessible"]:
            self.log(f"WebUI可访问: {self.webui_url}", "SUCCESS")
        else:
            self.log("WebUI不可访问", "ERROR")

        # 测试3: 插件加载
        self.log("\n测试3: 检查插件加载")
        results["plugin_loaded"] = self.test_plugin_loaded()
        if results["plugin_loaded"]:
            self.log("插件已加载", "SUCCESS")
        else:
            self.log("插件未加载", "WARN")

        # 显示容器日志
        self.log("\n最近日志:")
        print("-" * 50)
        logs = self.get_container_logs(20)
        print(logs)
        print("-" * 50)

        # 总结
        self.log("\n测试总结:", "INFO")
        passed = sum(results.values())
        total = len(results)
        self.log(f"通过: {passed}/{total}", "SUCCESS")

        return results


def main():
    """主函数"""
    print("======================================")
    print("Iris Memory Docker 环境测试")
    print("======================================\n")

    tester = DockerTester()
    results = tester.run_test()

    # 退出码
    exit_code = 0 if all(results.values()) else 1
    exit(exit_code)


if __name__ == "__main__":
    main()
