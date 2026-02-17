"""测试图片分析器的过滤功能"""
import pytest
from datetime import datetime
from iris_memory.multimodal import ImageAnalyzer, ImageAnalysisLevel, ImageInfo
from iris_memory.core.defaults import DEFAULTS


class MockContext:
    """Mock AstrBot context"""
    def get_using_provider(self, umo=''):
        return None


@pytest.fixture
def analyzer():
    """创建配置好的图片分析器"""
    return ImageAnalyzer(MockContext(), {
        'enable_image_analysis': DEFAULTS.image_analysis.enable_image_analysis,
        'daily_analysis_budget': 5,  # 测试用的小值
        'session_analysis_budget': 2,  # 测试用的小值
        'similar_image_window': DEFAULTS.image_analysis.similar_image_window,
        'recent_image_limit': DEFAULTS.image_analysis.recent_image_limit,
        'require_context_relevance': DEFAULTS.image_analysis.require_context_relevance
    })


class TestBudgetCheck:
    """测试预算检查功能"""
    
    def test_initial_budget_available(self, analyzer):
        """初始状态预算应该可用"""
        assert analyzer._check_budget("user1", "session1") is True
    
    def test_session_budget_exceeded(self, analyzer):
        """会话预算超限后应该返回False"""
        # 使用3次（超过限制2）
        for _ in range(3):
            analyzer._increment_budget('user1', 'session1')
        
        assert analyzer._check_budget("user1", "session1") is False
    
    def test_new_session_has_budget(self, analyzer):
        """新会话应该有独立预算"""
        # 耗尽session1的预算
        for _ in range(3):
            analyzer._increment_budget('user1', 'session1')
        
        # session2应该仍然有预算
        assert analyzer._check_budget("user1", "session2") is True
    
    def test_daily_budget_exceeded(self, analyzer):
        """每日预算超限后应该返回False"""
        # 使用6次（超过每日限制5）
        for i in range(6):
            analyzer._increment_budget('user1', f'session{i}')
        
        assert analyzer._check_budget("user1", "new_session") is False

    def test_daily_budget_override_is_used(self, analyzer):
        """调用时传入的每日预算应覆盖默认预算"""
        for i in range(3):
            analyzer._increment_budget('user1', f'session{i}')

        # 默认预算=5，当前用量=3，默认应可用
        assert analyzer._check_budget("user1", "new_session") is True
        # 覆盖预算=3，则应判定耗尽
        assert analyzer._check_budget("user1", "new_session", daily_analysis_budget=3) is False


class TestSimilarImageDetection:
    """测试相似图片检测功能"""
    
    def test_detect_same_hash(self, analyzer):
        """应该检测到相同哈希的图片"""
        analyzer._add_recent_image('hash1')
        assert analyzer._is_similar_recent_image('hash1') is True
    
    def test_new_hash_not_similar(self, analyzer):
        """新哈希应该不被认为是相似的"""
        analyzer._add_recent_image('hash1')
        assert analyzer._is_similar_recent_image('hash2') is False
    
    def test_empty_hash(self, analyzer):
        """空哈希应该返回False"""
        assert analyzer._is_similar_recent_image(None) is False
        assert analyzer._is_similar_recent_image('') is False


class TestContextRelevance:
    """测试上下文相关性检测"""
    
    @pytest.fixture
    def photo(self):
        return ImageInfo(url='http://example.com/photo.jpg', file='', is_sticker=False)
    
    @pytest.fixture
    def sticker(self):
        return ImageInfo(url='http://example.com/sticker.gif', file='', is_sticker=True)
    
    def test_question_context_relevant(self, analyzer, photo):
        """询问性文字应该被认为相关"""
        assert analyzer._check_context_relevance("这是什么", photo) is True
    
    def test_sharing_context_relevant(self, analyzer, photo):
        """分享类文字应该被认为相关"""
        assert analyzer._check_context_relevance("给你看看我的照片", photo) is True
    
    def test_empty_context_not_relevant(self, analyzer, photo):
        """空文字在require_context_relevance=True时应该不相关"""
        assert analyzer._check_context_relevance("", photo) is False
    
    def test_sticker_without_context_not_relevant(self, analyzer, sticker):
        """没有伴随文字的表情包不相关"""
        assert analyzer._check_context_relevance("", sticker) is False
    
    def test_emotion_context_relevant(self, analyzer, photo):
        """情感类文字应该被认为相关"""
        assert analyzer._check_context_relevance("好看吗", photo) is True


class TestStatistics:
    """测试统计信息功能"""
    
    def test_statistics_has_new_fields(self, analyzer):
        """统计信息应该包含新字段"""
        stats = analyzer.get_statistics()
        
        assert 'daily_analysis_count' in stats
        assert 'daily_budget_remaining' in stats
        assert 'recent_images_tracked' in stats
        assert 'active_sessions' in stats
        assert 'budget_exceeded' in stats
        assert 'similar_skipped' in stats
        assert 'context_skipped' in stats
    
    def test_budget_status(self, analyzer):
        """预算状态应该正确返回"""
        # 使用一些预算
        analyzer._increment_budget('user1', 'session1')
        
        status = analyzer.get_budget_status('session1')
        
        assert status['daily_used'] == 1
        assert status['daily_budget'] == 5
        assert status['daily_remaining'] == 4
        assert status['session_used'] == 1
        assert status['session_budget'] == 2
        assert status['session_remaining'] == 1


class TestClearMethods:
    """测试清理方法"""
    
    def test_clear_cache_clears_recent_images(self, analyzer):
        """清除缓存应该同时清除最近图片"""
        analyzer._add_recent_image('hash1')
        assert len(analyzer._recent_images) > 0
        
        analyzer.clear_cache()
        assert len(analyzer._recent_images) == 0
    
    def test_clear_all_budgets(self, analyzer):
        """应该能清除所有预算"""
        analyzer._increment_budget('user1', 'session1')
        analyzer.clear_all_budgets()
        
        assert len(analyzer._daily_analysis_count) == 0
        assert len(analyzer._session_analysis_count) == 0
    
    def test_reset_session_budget(self, analyzer):
        """应该能重置单个会话预算"""
        analyzer._increment_budget('user1', 'session1')
        analyzer._increment_budget('user1', 'session2')
        
        analyzer.reset_session_budget('session1')
        
        assert 'session1' not in analyzer._session_analysis_count
        assert analyzer._session_analysis_count.get('session2') == 1
