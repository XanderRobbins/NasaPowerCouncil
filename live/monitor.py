"""
System monitoring and health checks.
"""
import psutil
import pandas as pd
from datetime import datetime
from loguru import logger
from typing import Dict


class SystemMonitor:
    """
    Monitor system health and performance.
    """
    
    def __init__(self):
        self.checks = []
        
    def check_data_freshness(self, last_update: datetime, max_age_hours: int = 24) -> Dict:
        """Check if data is fresh."""
        age_hours = (datetime.now() - last_update).total_seconds() / 3600
        
        is_fresh = age_hours < max_age_hours
        
        return {
            'check': 'data_freshness',
            'passed': is_fresh,
            'age_hours': age_hours,
            'max_age_hours': max_age_hours
        }
    
    def check_system_resources(self) -> Dict:
        """Check CPU and memory usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        cpu_ok = cpu_percent < 80
        memory_ok = memory.percent < 80
        
        return {
            'check': 'system_resources',
            'passed': cpu_ok and memory_ok,
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent
        }
    
    def check_model_performance(self, recent_sharpe: float, threshold: float = 0.5) -> Dict:
        """Check if model is performing adequately."""
        is_ok = recent_sharpe > threshold
        
        return {
            'check': 'model_performance',
            'passed': is_ok,
            'recent_sharpe': recent_sharpe,
            'threshold': threshold
        }
    
    def run_all_checks(self, context: Dict) -> Dict:
        """Run all health checks."""
        results = []
        
        # Data freshness
        if 'last_update' in context:
            results.append(self.check_data_freshness(context['last_update']))
        
        # System resources
        results.append(self.check_system_resources())
        
        # Model performance
        if 'recent_sharpe' in context:
            results.append(self.check_model_performance(context['recent_sharpe']))
        
        # Overall status
        all_passed = all(r['passed'] for r in results)
        
        return {
            'timestamp': datetime.now(),
            'overall_status': 'HEALTHY' if all_passed else 'WARNING',
            'checks': results
        }
    
    def log_status(self, status: Dict):
        """Log system status."""
        if status['overall_status'] == 'HEALTHY':
            logger.info(f"✓ System health check: {status['overall_status']}")
        else:
            logger.warning(f"⚠ System health check: {status['overall_status']}")
            
            for check in status['checks']:
                if not check['passed']:
                    logger.warning(f"  - {check['check']} FAILED: {check}")