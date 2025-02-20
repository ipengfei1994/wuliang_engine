import hashlib
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from datetime import datetime
import json
import logging
from typing import Dict, List, Optional, Union, Any
import os
import uuid

class DataSecurity:
    def __init__(self, secret_key: Optional[str] = None):
        """初始化数据安全管理器"""
        self.secret_key = secret_key or self._generate_key()
        self.cipher_suite = Fernet(self._get_key())
        self.access_log = []
        self._setup_logging()

    def _setup_logging(self) -> None:
        """配置日志系统"""
        logging.basicConfig(
            filename='security.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _generate_key(self) -> bytes:
        """生成加密密钥"""
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(os.urandom(32)))
        return key

    def _get_key(self) -> bytes:
        """获取加密密钥"""
        if isinstance(self.secret_key, str):
            return self.secret_key.encode()
        return self.secret_key

    def encrypt_data(self, data: Union[str, bytes, Dict, List]) -> str:
        """加密数据"""
        if isinstance(data, (dict, list)):
            data = json.dumps(data)
        if isinstance(data, str):
            data = data.encode()
        
        encrypted_data = self.cipher_suite.encrypt(data)
        self._log_security_event('encrypt', {'data_type': type(data).__name__})
        return encrypted_data.decode()

    def decrypt_data(self, encrypted_data: str) -> Union[str, Dict, List]:
        """解密数据"""
        try:
            decrypted_data = self.cipher_suite.decrypt(encrypted_data.encode())
            try:
                # 尝试解析JSON
                result = json.loads(decrypted_data)
            except json.JSONDecodeError:
                result = decrypted_data.decode()
                
            self._log_security_event('decrypt', {'success': True})
            return result
        except Exception as e:
            self._log_security_event('decrypt', {'success': False, 'error': str(e)})
            raise ValueError("解密失败，可能是密钥不正确或数据已损坏")

    def hash_password(self, password: str) -> str:
        """密码哈希"""
        salt = os.urandom(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt,
            100000
        )
        return base64.b64encode(salt + hashed).decode()

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """验证密码"""
        try:
            decoded = base64.b64decode(hashed_password.encode())
            salt = decoded[:16]
            stored_hash = decoded[16:]
            
            hash_check = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode(),
                salt,
                100000
            )
            return hash_check == stored_hash
        except Exception:
            return False

    def generate_access_token(self, user_id: str, 
                            expiration_hours: int = 24) -> Dict[str, str]:
        """生成访问令牌"""
        token = str(uuid.uuid4())
        expiration = datetime.now().timestamp() + (expiration_hours * 3600)
        
        token_data = {
            'token': token,
            'user_id': user_id,
            'expiration': expiration
        }
        
        encrypted_token = self.encrypt_data(token_data)
        self._log_security_event('token_generated', {
            'user_id': user_id,
            'expiration_hours': expiration_hours
        })
        
        return {
            'access_token': token,
            'encrypted_token': encrypted_token
        }

    def validate_access_token(self, encrypted_token: str) -> Dict[str, Any]:
        """验证访问令牌"""
        try:
            token_data = self.decrypt_data(encrypted_token)
            current_time = datetime.now().timestamp()
            
            if token_data['expiration'] < current_time:
                raise ValueError("令牌已过期")
                
            self._log_security_event('token_validated', {
                'user_id': token_data['user_id'],
                'success': True
            })
            return token_data
        except Exception as e:
            self._log_security_event('token_validated', {
                'success': False,
                'error': str(e)
            })
            raise ValueError("无效的访问令牌")

    def _log_security_event(self, event_type: str, details: Dict) -> None:
        """记录安全事件"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details
        }
        self.access_log.append(event)
        logging.info(f"Security Event: {json.dumps(event)}")

    def get_security_audit_log(self, 
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None,
                             event_type: Optional[str] = None) -> List[Dict]:
        """获取安全审计日志"""
        filtered_logs = self.access_log
        
        if start_time:
            filtered_logs = [
                log for log in filtered_logs
                if datetime.fromisoformat(log['timestamp']) >= start_time
            ]
            
        if end_time:
            filtered_logs = [
                log for log in filtered_logs
                if datetime.fromisoformat(log['timestamp']) <= end_time
            ]
            
        if event_type:
            filtered_logs = [
                log for log in filtered_logs
                if log['event_type'] == event_type
            ]
            
        return filtered_logs

    def export_security_report(self, filepath: str) -> None:
        """导出安全报告"""
        report = {
            'total_events': len(self.access_log),
            'event_types': {},
            'recent_events': self.access_log[-10:],
            'timestamp': datetime.now().isoformat()
        }
        
        # 统计事件类型
        for event in self.access_log:
            event_type = event['event_type']
            if event_type in report['event_types']:
                report['event_types'][event_type] += 1
            else:
                report['event_types'][event_type] = 1
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)