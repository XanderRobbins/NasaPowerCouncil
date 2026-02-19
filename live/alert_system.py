"""
Alert and notification system.
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional
from datetime import datetime
from loguru import logger
import os


class AlertLevel:
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlertSystem:
    """
    Send alerts via email, SMS, or other channels.
    """
    
    def __init__(self,
                 email_enabled: bool = True,
                 email_recipients: Optional[List[str]] = None,
                 smtp_server: str = "smtp.gmail.com",
                 smtp_port: int = 587):
        self.email_enabled = email_enabled
        self.email_recipients = email_recipients or []
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        
        # Get credentials from environment
        self.smtp_username = os.getenv('ALERT_EMAIL_USERNAME')
        self.smtp_password = os.getenv('ALERT_EMAIL_PASSWORD')
        
        if self.email_enabled and not (self.smtp_username and self.smtp_password):
            logger.warning("Email alerts enabled but credentials not configured")
            self.email_enabled = False
    
    def send_alert(self, 
                   level: str,
                   title: str,
                   message: str,
                   include_timestamp: bool = True):
        """
        Send an alert.
        
        Args:
            level: Alert level (INFO, WARNING, ERROR, CRITICAL)
            title: Alert title
            message: Alert message
            include_timestamp: Whether to include timestamp
        """
        if include_timestamp:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            full_message = f"[{timestamp}] [{level}] {title}\n\n{message}"
        else:
            full_message = f"[{level}] {title}\n\n{message}"
        
        # Log
        if level == AlertLevel.CRITICAL or level == AlertLevel.ERROR:
            logger.error(full_message)
        elif level == AlertLevel.WARNING:
            logger.warning(full_message)
        else:
            logger.info(full_message)
        
        # Send via configured channels
        if self.email_enabled and self.email_recipients:
            self._send_email(level, title, full_message)
    
    def _send_email(self, level: str, subject: str, body: str):
        """Send email alert."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_username
            msg['To'] = ', '.join(self.email_recipients)
            msg['Subject'] = f"[{level}] Climate Futures Alert: {subject}"
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            logger.debug(f"Email alert sent to {len(self.email_recipients)} recipients")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def alert_data_issue(self, commodity: str, issue: str):
        """Alert about data quality issue."""
        self.send_alert(
            AlertLevel.WARNING,
            f"Data Issue: {commodity}",
            f"Data quality issue detected for {commodity}:\n\n{issue}"
        )
    
    def alert_model_degradation(self, commodity: str, recent_sharpe: float):
        """Alert about model performance degradation."""
        self.send_alert(
            AlertLevel.WARNING,
            f"Model Performance: {commodity}",
            f"Model performance degradation detected for {commodity}.\n"
            f"Recent Sharpe ratio: {recent_sharpe:.2f}"
        )
    
    def alert_risk_breach(self, breach_type: str, details: str):
        """Alert about risk limit breach."""
        self.send_alert(
            AlertLevel.ERROR,
            f"Risk Breach: {breach_type}",
            f"Risk limit breached:\n\n{details}"
        )
    
    def alert_system_error(self, error: str):
        """Alert about system error."""
        self.send_alert(
            AlertLevel.CRITICAL,
            "System Error",
            f"Critical system error:\n\n{error}"
        )
    
    def alert_trade_execution(self, orders: List, execution_details: dict):
        """Alert about trade execution."""
        message = "Trades executed:\n\n"
        for order in orders:
            message += f"- {order.side.value} {order.quantity:.2f} {order.commodity} @ {order.contract}\n"
        
        message += f"\nTotal transaction costs: ${execution_details.get('total_cost', 0):.2f}"
        
        self.send_alert(
            AlertLevel.INFO,
            "Trade Execution",
            message
        )
    
    def daily_summary(self, portfolio_value: float, daily_pnl: float, positions: dict):
        """Send daily performance summary."""
        message = f"""
Daily Performance Summary
========================

Portfolio Value: ${portfolio_value:,.2f}
Daily P&L: ${daily_pnl:,.2f} ({(daily_pnl/portfolio_value)*100:.2f}%)

Current Positions:
"""
        for commodity, position in positions.items():
            if abs(position) > 0.01:
                message += f"  {commodity}: {position:.2f} contracts\n"
        
        self.send_alert(
            AlertLevel.INFO,
            "Daily Summary",
            message
        )


# Global alert instance
alert_system = AlertSystem(
    email_enabled=False,  # Set to True and configure in production
    email_recipients=[]
)