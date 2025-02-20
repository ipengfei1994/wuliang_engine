from flask import Flask, jsonify, request
from engine.social.analyzer import SocialMediaAnalyzer
from engine.content.management import ContentManager
from engine.user.behavior import UserBehaviorAnalyzer
from engine.operation.efficiency import OperationalEfficiencyAnalyzer
from engine.operation.ad_performance import AdPerformanceAnalyzer
from utils.system.config import ConfigManager
from utils.system.logger import LogManager

app = Flask(__name__)

# 初始化配置和日志
config_manager = ConfigManager()
log_manager = LogManager()

# 初始化分析器
social_media = SocialMediaAnalyzer()
content = ContentManager()
user = UserBehaviorAnalyzer()
operation = OperationalEfficiencyAnalyzer()
ad_performance = AdPerformanceAnalyzer()

# 全局异常处理
@app.errorhandler(Exception)
def handle_error(error):
    log_manager.log_error(error, {"path": request.path})
    return jsonify({"status": "error", "message": str(error)}), 500

# 路由注册
from api.routes import register_routes
register_routes(app)

if __name__ == "__main__":
    config = config_manager.get_config('api')
    app.run(
        host=config.get('host', '0.0.0.0'),
        port=config.get('port', 8000),
        debug=config.get('debug', False)
    )