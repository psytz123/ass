"""
Webhook Management System for AI Orchestration Framework
Handles webhook registration, event triggering, and notifications
"""

import asyncio
import json
import hashlib
import hmac
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import uuid

from models import db

logger = logging.getLogger(__name__)


class WebhookEvent(Enum):
    """Available webhook events"""
    REQUEST_COMPLETED = "request.completed"
    REQUEST_FAILED = "request.failed"
    CONSENSUS_ACHIEVED = "consensus.achieved"
    BUDGET_EXCEEDED = "budget.exceeded"
    AB_TEST_COMPLETED = "ab_test.completed"
    PERFORMANCE_DEGRADED = "performance.degraded"
    USER_MILESTONE = "user.milestone"


@dataclass
class WebhookPayload:
    """Standard webhook payload structure"""
    event: str
    timestamp: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class WebhookManager:
    """Manages webhook registrations and notifications"""
    
    def __init__(self):
        self.retry_attempts = 3
        self.retry_delay = 5  # seconds
        self.timeout = 10  # seconds
        self._active_webhooks_cache = {}
        self._last_cache_update = None
        self._cache_ttl = 300  # 5 minutes
        
    def register_webhook(self, url: str, events: List[str], secret: Optional[str] = None,
                        description: Optional[str] = None) -> Dict[str, Any]:
        """Register a new webhook"""
        from models import Webhook
        
        # Validate URL
        if not self._validate_url(url):
            raise ValueError("Invalid webhook URL")
            
        # Validate events
        valid_events = [e.value for e in WebhookEvent]
        invalid_events = [e for e in events if e not in valid_events]
        if invalid_events:
            raise ValueError(f"Invalid events: {invalid_events}")
            
        # Generate webhook ID
        webhook_id = str(uuid.uuid4())
        
        # Create webhook record
        webhook = Webhook(
            webhook_id=webhook_id,
            url=url,
            events=json.dumps(events),
            secret=secret,
            description=description,
            active=True,
            created_at=datetime.utcnow()
        )
        
        db.session.add(webhook)
        db.session.commit()
        
        # Clear cache
        self._clear_cache()
        
        return {
            'webhook_id': webhook_id,
            'url': url,
            'events': events,
            'active': True,
            'created_at': webhook.created_at.isoformat()
        }
        
    def update_webhook(self, webhook_id: str, url: Optional[str] = None,
                      events: Optional[List[str]] = None, secret: Optional[str] = None,
                      active: Optional[bool] = None) -> Dict[str, Any]:
        """Update webhook configuration"""
        from models import Webhook
        
        webhook = Webhook.query.filter_by(webhook_id=webhook_id).first()
        if not webhook:
            raise ValueError("Webhook not found")
            
        if url is not None:
            if not self._validate_url(url):
                raise ValueError("Invalid webhook URL")
            webhook.url = url
            
        if events is not None:
            valid_events = [e.value for e in WebhookEvent]
            invalid_events = [e for e in events if e not in valid_events]
            if invalid_events:
                raise ValueError(f"Invalid events: {invalid_events}")
            webhook.events = json.dumps(events)
            
        if secret is not None:
            webhook.secret = secret
            
        if active is not None:
            webhook.active = active
            
        webhook.updated_at = datetime.utcnow()
        db.session.commit()
        
        # Clear cache
        self._clear_cache()
        
        return self.get_webhook(webhook_id)
        
    def delete_webhook(self, webhook_id: str) -> bool:
        """Delete a webhook"""
        from models import Webhook
        
        webhook = Webhook.query.filter_by(webhook_id=webhook_id).first()
        if not webhook:
            raise ValueError("Webhook not found")
            
        db.session.delete(webhook)
        db.session.commit()
        
        # Clear cache
        self._clear_cache()
        
        return True
        
    def get_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """Get webhook details"""
        from models import Webhook
        
        webhook = Webhook.query.filter_by(webhook_id=webhook_id).first()
        if not webhook:
            raise ValueError("Webhook not found")
            
        return {
            'webhook_id': webhook.webhook_id,
            'url': webhook.url,
            'events': json.loads(webhook.events),
            'active': webhook.active,
            'description': webhook.description,
            'created_at': webhook.created_at.isoformat(),
            'last_triggered': webhook.last_triggered.isoformat() if webhook.last_triggered else None,
            'trigger_count': webhook.trigger_count
        }
        
    def list_webhooks(self) -> List[Dict[str, Any]]:
        """List all registered webhooks"""
        from models import Webhook
        
        webhooks = Webhook.query.all()
        return [self.get_webhook(w.webhook_id) for w in webhooks]
        
    async def trigger_event(self, event: WebhookEvent, data: Dict[str, Any],
                          metadata: Optional[Dict[str, Any]] = None):
        """Trigger webhook event for all subscribed webhooks"""
        # Get active webhooks for this event
        webhooks = self._get_active_webhooks(event.value)
        
        if not webhooks:
            return
            
        # Create payload
        payload = WebhookPayload(
            event=event.value,
            timestamp=datetime.utcnow().isoformat(),
            data=data,
            metadata=metadata or {}
        )
        
        # Send to all webhooks asynchronously
        tasks = []
        for webhook in webhooks:
            task = asyncio.create_task(
                self._send_webhook(webhook, payload)
            )
            tasks.append(task)
            
        # Wait for all webhooks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _send_webhook(self, webhook: Dict[str, Any], payload: WebhookPayload):
        """Send webhook notification with retry logic"""
        url = webhook['url']
        secret = webhook.get('secret')
        webhook_id = webhook['webhook_id']
        
        # Prepare request
        headers = {
            'Content-Type': 'application/json',
            'X-Webhook-Event': payload.event,
            'X-Webhook-ID': webhook_id,
            'X-Webhook-Timestamp': payload.timestamp
        }
        
        # Convert payload to JSON
        payload_json = json.dumps({
            'event': payload.event,
            'timestamp': payload.timestamp,
            'data': payload.data,
            'metadata': payload.metadata
        })
        
        # Add signature if secret provided
        if secret:
            signature = self._generate_signature(payload_json, secret)
            headers['X-Webhook-Signature'] = signature
            
        # Attempt delivery with retries
        for attempt in range(self.retry_attempts):
            try:
                response = await asyncio.to_thread(
                    requests.post,
                    url,
                    data=payload_json,
                    headers=headers,
                    timeout=self.timeout
                )
                
                if response.status_code >= 200 and response.status_code < 300:
                    # Success
                    await self._update_webhook_stats(webhook_id, success=True)
                    logger.info(f"Webhook delivered successfully to {url}")
                    return
                else:
                    logger.warning(f"Webhook delivery failed to {url}: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Webhook delivery error to {url}: {str(e)}")
                
            # Wait before retry
            if attempt < self.retry_attempts - 1:
                await asyncio.sleep(self.retry_delay)
                
        # All attempts failed
        await self._update_webhook_stats(webhook_id, success=False)
        logger.error(f"Webhook delivery failed after {self.retry_attempts} attempts to {url}")
        
    async def _update_webhook_stats(self, webhook_id: str, success: bool):
        """Update webhook statistics"""
        from models import Webhook
        
        try:
            webhook = Webhook.query.filter_by(webhook_id=webhook_id).first()
            if webhook:
                webhook.last_triggered = datetime.utcnow()
                webhook.trigger_count += 1
                if not success:
                    webhook.failure_count += 1
                db.session.commit()
        except Exception as e:
            logger.error(f"Failed to update webhook stats: {str(e)}")
            
    def _get_active_webhooks(self, event: str) -> List[Dict[str, Any]]:
        """Get active webhooks for a specific event"""
        # Check cache
        if self._active_webhooks_cache and self._last_cache_update:
            cache_age = (datetime.utcnow() - self._last_cache_update).total_seconds()
            if cache_age < self._cache_ttl:
                return self._active_webhooks_cache.get(event, [])
                
        # Rebuild cache
        self._rebuild_cache()
        
        return self._active_webhooks_cache.get(event, [])
        
    def _rebuild_cache(self):
        """Rebuild active webhooks cache"""
        from models import Webhook
        
        self._active_webhooks_cache = {}
        
        webhooks = Webhook.query.filter_by(active=True).all()
        for webhook in webhooks:
            events = json.loads(webhook.events)
            webhook_data = {
                'webhook_id': webhook.webhook_id,
                'url': webhook.url,
                'secret': webhook.secret
            }
            
            for event in events:
                if event not in self._active_webhooks_cache:
                    self._active_webhooks_cache[event] = []
                self._active_webhooks_cache[event].append(webhook_data)
                
        self._last_cache_update = datetime.utcnow()
        
    def _clear_cache(self):
        """Clear webhook cache"""
        self._active_webhooks_cache = {}
        self._last_cache_update = None
        
    def _validate_url(self, url: str) -> bool:
        """Validate webhook URL"""
        try:
            # Basic URL validation
            if not url.startswith(('http://', 'https://')):
                return False
            # Additional validation could be added here
            return True
        except:
            return False
            
    def _generate_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC signature for webhook payload"""
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
        
    async def test_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """Send a test event to a webhook"""
        webhook = self.get_webhook(webhook_id)
        
        # Create test payload
        test_data = {
            'test': True,
            'webhook_id': webhook_id,
            'message': 'This is a test webhook event'
        }
        
        payload = WebhookPayload(
            event='webhook.test',
            timestamp=datetime.utcnow().isoformat(),
            data=test_data,
            metadata={'triggered_by': 'manual_test'}
        )
        
        # Send test webhook
        try:
            await self._send_webhook(webhook, payload)
            return {
                'success': True,
                'message': 'Test event sent successfully',
                'webhook_id': webhook_id
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Test event failed: {str(e)}',
                'webhook_id': webhook_id
            }


# Global webhook manager instance
webhook_manager = WebhookManager()