import unittest
from engine.user.behavior import UserBehaviorAnalyzer

class TestUserBehaviorAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = UserBehaviorAnalyzer()

    def test_analyze_behavior(self):
        """测试用户行为分析功能"""
        user_id = "user101"
        result = self.analyzer.analyze_behavior(user_id)
        
        # 验证返回结果包含所需的关键字段
        self.assertIsInstance(result, dict)
        self.assertIn('total_actions', result)
        self.assertIn('unique_actions', result)
        self.assertIn('active_days', result)
        self.assertIn('avg_daily_actions', result)
        self.assertIn('first_action', result)
        self.assertIn('last_action', result)

    def test_get_segments(self):
        """测试用户分群功能"""
        segments = self.analyzer.get_segments()
        
        # 验证返回结果是列表且包含分群信息
        self.assertIsInstance(segments, list)
        if segments:
            self.assertIn('cluster_id', segments[0])
            self.assertIn('users', segments[0])

if __name__ == '__main__':
    unittest.main()