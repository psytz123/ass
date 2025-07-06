import asyncio
import logging
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, session
from datetime import datetime

from core.types import TaskComplexity
from config import Config

logger = logging.getLogger(__name__)

# Create blueprint
api_bp = Blueprint('api', __name__)

# Initialize framework lazily to avoid circular imports
framework = None

def get_framework():
    global framework
    if framework is None:
        from core.framework import AIOrchestrationFramework
        try:
            framework = AIOrchestrationFramework.from_config('config.yaml')
        except Exception as e:
            logger.warning(f"Failed to load config.yaml, using defaults: {e}")
            framework = AIOrchestrationFramework(Config())
    return framework

@api_bp.route('/')
def dashboard():
    """Main dashboard"""
    return render_template('dashboard.html')

@api_bp.route('/test')
def test_framework():
    """Test framework interface"""
    try:
        fw = get_framework()
        available_providers = fw.get_available_providers()
        supported_tasks = fw.get_supported_task_types()
    except Exception as e:
        logger.error(f"Failed to get providers: {e}")
        available_providers = ['openai', 'anthropic']
        supported_tasks = ['general', 'analysis', 'creative', 'factual']
    
    return render_template(
        'test_framework.html',
        providers=available_providers,
        task_types=supported_tasks
    )

@api_bp.route('/performance')
def performance():
    """Performance metrics page"""
    return render_template('performance.html')

@api_bp.route('/api/process', methods=['POST'])
def process_request():
    """Process AI request"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        task_type = data.get('task_type', 'general')
        user_id = data.get('user_id', 'anonymous')
        require_consensus = data.get('require_consensus')
        complexity_override = data.get('complexity_override')
        system_prompt = data.get('system_prompt')
        temperature = data.get('temperature')
        max_tokens = data.get('max_tokens')
        
        # Convert complexity override
        if complexity_override:
            try:
                complexity_override = TaskComplexity(complexity_override)
            except ValueError:
                complexity_override = None
        
        # Process request asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            fw = get_framework()
            result = loop.run_until_complete(
                fw.process_request(
                    prompt=prompt,
                    task_type=task_type,
                    user_id=user_id,
                    require_consensus=require_consensus,
                    complexity_override=complexity_override,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            )
        finally:
            loop.close()
        
        return jsonify({
            'success': True,
            'response': result.response,
            'metadata': result.metadata
        })
        
    except Exception as e:
        logger.error(f"API request failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/api/performance', methods=['GET'])
def get_performance():
    """Get performance metrics"""
    try:
        time_window = request.args.get('hours', 24, type=int)
        
        # Get metrics asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            fw = get_framework()
            metrics = loop.run_until_complete(
                fw.get_performance_metrics(time_window_hours=time_window)
            )
        finally:
            loop.close()
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'time_window_hours': time_window
        })
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/api/history/<user_id>', methods=['GET'])
def get_history(user_id):
    """Get conversation history"""
    try:
        limit = request.args.get('limit', 50, type=int)
        task_type = request.args.get('task_type')
        
        # Get history asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            fw = get_framework()
            history = loop.run_until_complete(
                fw.get_conversation_history(
                    user_id=user_id,
                    limit=limit,
                    task_type=task_type
                )
            )
        finally:
            loop.close()
        
        return jsonify({
            'success': True,
            'history': history,
            'user_id': user_id
        })
        
    except Exception as e:
        logger.error(f"Failed to get conversation history: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/api/providers', methods=['GET'])
def get_providers():
    """Get available providers"""
    try:
        fw = get_framework()
        providers = fw.get_available_providers()
        task_types = fw.get_supported_task_types()
        
        return jsonify({
            'success': True,
            'providers': providers,
            'task_types': task_types
        })
        
    except Exception as e:
        logger.error(f"Failed to get providers: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Form-based routes for better UX
@api_bp.route('/submit_test', methods=['POST'])
def submit_test():
    """Submit test request via form"""
    try:
        prompt = request.form.get('prompt', '').strip()
        if not prompt:
            flash('Prompt is required', 'error')
            return redirect(url_for('api.test_framework'))
        
        task_type = request.form.get('task_type', 'general')
        user_id = request.form.get('user_id', 'web_user')
        require_consensus = request.form.get('require_consensus') == 'on'
        
        # Process request
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            fw = get_framework()
            result = loop.run_until_complete(
                fw.process_request(
                    prompt=prompt,
                    task_type=task_type,
                    user_id=user_id,
                    require_consensus=require_consensus
                )
            )
        finally:
            loop.close()
        
        flash('Request processed successfully!', 'success')
        fw = get_framework()
        return render_template(
            'test_framework.html',
            providers=fw.get_available_providers(),
            task_types=fw.get_supported_task_types(),
            result=result,
            last_prompt=prompt
        )
        
    except Exception as e:
        logger.error(f"Form submission failed: {e}")
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('api.test_framework'))


@api_bp.route('/api/batch', methods=['POST'])
def batch_process():
    """Batch process multiple AI requests"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        requests = data.get('requests', [])
        if not requests:
            return jsonify({'error': 'No requests provided'}), 400
        
        if len(requests) > 50:  # Limit batch size
            return jsonify({'error': 'Batch size exceeds maximum of 50 requests'}), 400
        
        user_id = data.get('user_id', 'anonymous')
        
        # Process requests asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            fw = get_framework()
            
            async def process_batch():
                results = []
                errors = []
                
                # Process requests in chunks for better performance
                chunk_size = 5
                for i in range(0, len(requests), chunk_size):
                    chunk = requests[i:i + chunk_size]
                    
                    # Create tasks for concurrent processing
                    tasks = []
                    for req in chunk:
                        prompt = req.get('prompt', '').strip()
                        if not prompt:
                            errors.append({'index': i + len(tasks), 'error': 'Prompt is required'})
                            continue
                        
                        task = fw.process_request(
                            prompt=prompt,
                            task_type=req.get('task_type', 'general'),
                            user_id=user_id,
                            require_consensus=req.get('require_consensus', False),
                            complexity_override=req.get('complexity_override'),
                            system_prompt=req.get('system_prompt'),
                            temperature=req.get('temperature'),
                            max_tokens=req.get('max_tokens')
                        )
                        tasks.append((i + len(tasks), task))
                    
                    # Process chunk
                    if tasks:
                        chunk_results = await asyncio.gather(
                            *[task for _, task in tasks], 
                            return_exceptions=True
                        )
                        
                        # Process results
                        for (idx, _), result in zip(tasks, chunk_results):
                            if isinstance(result, Exception):
                                errors.append({
                                    'index': idx,
                                    'error': str(result)
                                })
                            else:
                                results.append({
                                    'index': idx,
                                    'response': result.response,
                                    'metadata': result.metadata
                                })
                
                return results, errors
            
            results, errors = loop.run_until_complete(process_batch())
        finally:
            loop.close()
        
        return jsonify({
            'success': True,
            'total_processed': len(results) + len(errors),
            'successful': len(results),
            'failed': len(errors),
            'results': results,
            'errors': errors
        })
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/realtime-demo')
def realtime_demo():
    """Real-time updates and batch processing demo"""
    return render_template('realtime_demo.html')

@api_bp.route('/advanced-features')
def advanced_features():
    """Show advanced features dashboard"""
    try:
        fw = get_framework()
        
        # Run async operations in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Get performance analysis
            performance_analysis = loop.run_until_complete(
                fw.performance_optimizer.analyze_patterns(time_window_hours=168)
            )
            
            # Get sample user for context demonstration
            sample_user_id = session.get('demo_user_id', 'demo_user_001')
            context_info = loop.run_until_complete(
                fw.context_memory.get_relevant_context(
                    user_id=sample_user_id,
                    current_prompt="What is machine learning?",
                    task_type="educational"
                )
            )
        finally:
            loop.close()
        
        # Prepare feature data
        features_data = {
            'optimization': {
                'performance_scores': dict(fw.performance_optimizer.performance_scores),
                'pattern_analysis': performance_analysis,
                'exploration_rate': fw.adaptive_router.exploration_rate
            },
            'consensus': {
                'strategies': list(fw.consensus_engine.strategies.keys()),
                'default_strategy': fw.consensus_engine.default_strategy,
                'confidence_threshold': 0.6
            },
            'context_memory': {
                'conversation_count': context_info['conversation_count'],
                'context_patterns': context_info['context_patterns'],
                'user_preferences': dict(fw.context_memory.user_preferences)
            }
        }
        
        return render_template('advanced_features.html', features_data=features_data)
        
    except Exception as e:
        logger.error(f"Advanced features page error: {e}")
        return render_template('advanced_features.html', 
                             features_data=None, 
                             error=str(e))

# Cost Tracking API endpoints
@api_bp.route('/api/costs')
def get_costs():
    """Get cost metrics for API usage"""
    try:
        fw = get_framework()
        hours = int(request.args.get('hours', 24))
        user_id = request.args.get('user_id', None)
        
        # Run async operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            cost_metrics = loop.run_until_complete(
                fw.cost_tracker.get_cost_metrics(
                    time_window_hours=hours,
                    user_id=user_id
                )
            )
            
            # Get budget alerts
            daily_budget = float(request.args.get('daily_budget', 10.0))
            monthly_budget = float(request.args.get('monthly_budget', 300.0))
            
            budget_alerts = loop.run_until_complete(
                fw.cost_tracker.get_budget_alerts(
                    daily_budget=daily_budget,
                    monthly_budget=monthly_budget
                )
            )
            
            cost_metrics['budget_alerts'] = budget_alerts
            
        finally:
            loop.close()
        
        return jsonify({
            'success': True,
            'costs': cost_metrics
        })
        
    except Exception as e:
        logger.error(f"Failed to get cost metrics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@api_bp.route('/costs')
def costs_dashboard():
    """Cost tracking dashboard"""
    return render_template('costs.html')

# A/B Testing API endpoints
@api_bp.route('/api/ab-tests', methods=['GET', 'POST'])
def manage_ab_tests():
    """Manage A/B tests"""
    try:
        fw = get_framework()
        
        if request.method == 'POST':
            # Create new A/B test
            data = request.json
            
            # Create test groups
            from core.ab_testing import ABTestGroup, RoutingStrategy
            
            groups = []
            for group_data in data['groups']:
                strategy = RoutingStrategy(group_data['strategy'])
                group = ABTestGroup(
                    name=group_data['name'],
                    strategy=strategy,
                    config=group_data.get('config', {}),
                    allocation_percentage=group_data['allocation']
                )
                groups.append(group)
            
            result = fw.ab_testing.create_test(
                test_id=data['test_id'],
                name=data['name'],
                groups=groups,
                duration_hours=data.get('duration_hours', 24),
                min_samples_per_group=data.get('min_samples', 100)
            )
            
            return jsonify({
                'success': True,
                'test': result
            })
        
        else:
            # Get test results
            test_id = request.args.get('test_id')
            include_active = request.args.get('include_active', 'true').lower() == 'true'
            
            # Run async operation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                results = loop.run_until_complete(
                    fw.ab_testing.get_test_results(
                        test_id=test_id,
                        include_active=include_active
                    )
                )
            finally:
                loop.close()
            
            return jsonify({
                'success': True,
                'results': results
            })
            
    except Exception as e:
        logger.error(f"A/B testing error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@api_bp.route('/api/ab-tests/<test_id>/stop', methods=['POST'])
def stop_ab_test(test_id):
    """Stop an active A/B test"""
    try:
        fw = get_framework()
        result = fw.ab_testing.stop_test(test_id)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Failed to stop A/B test: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@api_bp.route('/ab-testing')
def ab_testing_dashboard():
    """A/B testing dashboard"""
    return render_template('ab_testing.html')

# Usage Analytics API endpoints
@api_bp.route('/api/usage-analytics')
def get_usage_analytics():
    """Get comprehensive usage analytics"""
    try:
        fw = get_framework()
        hours = int(request.args.get('hours', 168))  # Default to 1 week
        user_id = request.args.get('user_id', None)
        
        # Run async operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            analytics = loop.run_until_complete(
                fw.usage_analytics.get_usage_insights(
                    time_window_hours=hours,
                    user_id=user_id
                )
            )
        finally:
            loop.close()
        
        return jsonify({
            'success': True,
            'analytics': analytics
        })
        
    except Exception as e:
        logger.error(f"Failed to get usage analytics: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@api_bp.route('/usage-analytics')
def usage_analytics_dashboard():
    """Usage analytics dashboard"""
    return render_template('usage_analytics.html')


# Webhook endpoints
@api_bp.route('/api/webhooks', methods=['GET'])
def list_webhooks():
    """Get list of registered webhooks"""
    try:
        from core.webhooks import webhook_manager
        webhooks = webhook_manager.list_webhooks()
        return jsonify({'success': True, 'webhooks': webhooks})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/api/webhooks', methods=['POST'])
def create_webhook():
    """Register a new webhook"""
    try:
        from core.webhooks import webhook_manager
        data = request.json
        
        webhook = webhook_manager.register_webhook(
            url=data['url'],
            events=data['events'],
            secret=data.get('secret'),
            description=data.get('description')
        )
        
        return jsonify({'success': True, 'webhook': webhook})
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/api/webhooks/<webhook_id>', methods=['GET'])
def get_webhook(webhook_id):
    """Get webhook details"""
    try:
        from core.webhooks import webhook_manager
        webhook = webhook_manager.get_webhook(webhook_id)
        return jsonify({'success': True, 'webhook': webhook})
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/api/webhooks/<webhook_id>', methods=['PUT'])
def update_webhook(webhook_id):
    """Update webhook configuration"""
    try:
        from core.webhooks import webhook_manager
        data = request.json
        
        webhook = webhook_manager.update_webhook(
            webhook_id=webhook_id,
            url=data.get('url'),
            events=data.get('events'),
            secret=data.get('secret'),
            active=data.get('active')
        )
        
        return jsonify({'success': True, 'webhook': webhook})
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/api/webhooks/<webhook_id>', methods=['DELETE'])
def delete_webhook(webhook_id):
    """Delete a webhook"""
    try:
        from core.webhooks import webhook_manager
        webhook_manager.delete_webhook(webhook_id)
        return jsonify({'success': True})
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/api/webhooks/<webhook_id>/test', methods=['POST'])
def test_webhook(webhook_id):
    """Send a test event to the webhook"""
    try:
        from core.webhooks import webhook_manager
        import asyncio
        
        # Run async function synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(webhook_manager.test_webhook(webhook_id))
        loop.close()
        
        return jsonify(result)
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@api_bp.route('/api/webhooks/events', methods=['GET'])
def list_webhook_events():
    """Get list of available webhook events"""
    from core.webhooks import WebhookEvent
    
    events = {
        event.value: f"Event triggered when {event.value.replace('.', ' ').replace('_', ' ')}"
        for event in WebhookEvent
    }
    
    return jsonify({'success': True, 'events': events})


# API Documentation endpoint
@api_bp.route('/api/docs')
def api_documentation():
    """Redirect to Swagger UI documentation"""
    return redirect('/api/v1/docs')
