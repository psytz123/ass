import asyncio
import logging
import json
from typing import Dict, List, Optional
from datetime import datetime
from flask_socketio import emit, join_room, leave_room
from flask import request

from api.socketio_init import socketio
from api.routes import get_framework
from app import app
from core.types import TaskComplexity
from app import db
from models import Conversation

logger = logging.getLogger(__name__)

# Store active sessions and their rooms
active_sessions: Dict[str, Dict] = {}

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    client_id = request.sid
    active_sessions[client_id] = {
        'connected_at': datetime.utcnow(),
        'user_id': request.args.get('user_id', 'anonymous'),
        'rooms': set()
    }
    logger.info(f"Client {client_id} connected")
    emit('connection_established', {'client_id': client_id})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    if client_id in active_sessions:
        # Leave all rooms
        for room in active_sessions[client_id]['rooms']:
            leave_room(room)
        del active_sessions[client_id]
        logger.info(f"Client {client_id} disconnected")

@socketio.on('join_updates')
def handle_join_updates(data):
    """Join real-time updates room"""
    client_id = request.sid
    room = data.get('room', 'general_updates')
    join_room(room)
    
    if client_id in active_sessions:
        active_sessions[client_id]['rooms'].add(room)
    
    emit('joined_room', {'room': room})
    logger.info(f"Client {client_id} joined room {room}")

@socketio.on('process_request')
def handle_process_request(data):
    """Process AI request with real-time updates"""
    client_id = request.sid
    
    try:
        # Validate input
        prompt = data.get('prompt', '').strip()
        if not prompt:
            emit('error', {'message': 'Prompt is required'})
            return
        
        task_type = data.get('task_type', 'general')
        user_id = data.get('user_id', active_sessions.get(client_id, {}).get('user_id', 'anonymous'))
        require_consensus = data.get('require_consensus', False)
        complexity_override = data.get('complexity_override')
        
        # Convert complexity override
        if complexity_override:
            try:
                complexity_override = TaskComplexity(complexity_override)
            except ValueError:
                complexity_override = None
        
        # Emit processing started
        emit('processing_started', {
            'prompt': prompt,
            'task_type': task_type,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Process request asynchronously
        socketio.start_background_task(
            process_request_async,
            client_id,
            prompt,
            task_type,
            user_id,
            require_consensus,
            complexity_override,
            data.get('system_prompt'),
            data.get('temperature'),
            data.get('max_tokens')
        )
        
    except Exception as e:
        logger.error(f"Error processing WebSocket request: {e}")
        emit('error', {'message': str(e)})

def process_request_async(client_id, prompt, task_type, user_id, require_consensus, 
                         complexity_override, system_prompt, temperature, max_tokens):
    """Process request asynchronously and emit updates"""
    try:
        with app.app_context():
            # Get framework
            framework = get_framework()
            
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Emit complexity evaluation started
                socketio.emit('stage_update', {
                    'stage': 'complexity_evaluation',
                    'message': 'Evaluating task complexity...'
                }, room=client_id)
                
                # Process request with progress updates
                async def process_with_updates():
                    # Start processing
                    result = await framework.process_request(
                        prompt=prompt,
                        task_type=task_type,
                        user_id=user_id,
                        require_consensus=require_consensus,
                        complexity_override=complexity_override,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    # Emit provider selection
                    if 'providers_used' in result.metadata:
                        socketio.emit('stage_update', {
                            'stage': 'provider_selection',
                            'message': f"Selected providers: {', '.join(result.metadata['providers_used'])}",
                            'providers': result.metadata['providers_used']
                        }, room=client_id)
                    
                    # Emit consensus stage if applicable
                    if 'consensus_confidence' in result.metadata:
                        socketio.emit('stage_update', {
                            'stage': 'consensus_calculation',
                            'message': f"Calculating consensus (confidence: {result.metadata['consensus_confidence']:.2f})",
                            'confidence': result.metadata['consensus_confidence']
                        }, room=client_id)
                    
                    return result
                
                result = loop.run_until_complete(process_with_updates())
                
                # Emit completion
                socketio.emit('processing_completed', {
                    'response': result.response,
                    'metadata': result.metadata,
                    'timestamp': datetime.utcnow().isoformat()
                }, room=client_id)
                
            finally:
                loop.close()
                
    except Exception as e:
        logger.error(f"Error in async processing: {e}")
        socketio.emit('error', {
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }, room=client_id)

@socketio.on('batch_process')
def handle_batch_process(data):
    """Handle batch processing of multiple requests"""
    client_id = request.sid
    
    try:
        # Validate input
        requests = data.get('requests', [])
        if not requests:
            emit('error', {'message': 'No requests provided'})
            return
        
        if len(requests) > 50:  # Limit batch size
            emit('error', {'message': 'Batch size exceeds maximum of 50 requests'})
            return
        
        user_id = data.get('user_id', active_sessions.get(client_id, {}).get('user_id', 'anonymous'))
        
        # Emit batch processing started
        emit('batch_started', {
            'total_requests': len(requests),
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Process batch asynchronously
        socketio.start_background_task(
            process_batch_async,
            client_id,
            requests,
            user_id
        )
        
    except Exception as e:
        logger.error(f"Error starting batch processing: {e}")
        emit('error', {'message': str(e)})

def process_batch_async(client_id, requests, user_id):
    """Process batch of requests asynchronously"""
    try:
        with app.app_context():
            framework = get_framework()
            
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                async def process_batch():
                    results = []
                    
                    # Process requests concurrently in chunks
                    chunk_size = 5  # Process 5 at a time
                    for i in range(0, len(requests), chunk_size):
                        chunk = requests[i:i + chunk_size]
                        
                        # Create tasks for concurrent processing
                        tasks = []
                        for idx, req in enumerate(chunk):
                            task = framework.process_request(
                                prompt=req.get('prompt', ''),
                                task_type=req.get('task_type', 'general'),
                                user_id=user_id,
                                require_consensus=req.get('require_consensus', False)
                            )
                            tasks.append(task)
                        
                        # Process chunk
                        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # Process results
                        for idx, result in enumerate(chunk_results):
                            req_idx = i + idx
                            
                            if isinstance(result, Exception):
                                socketio.emit('batch_item_error', {
                                    'index': req_idx,
                                    'error': str(result)
                                }, room=client_id)
                                results.append({
                                    'index': req_idx,
                                    'error': str(result)
                                })
                            else:
                                socketio.emit('batch_item_completed', {
                                    'index': req_idx,
                                    'response': result.response,
                                    'metadata': result.metadata
                                }, room=client_id)
                                results.append({
                                    'index': req_idx,
                                    'response': result.response,
                                    'metadata': result.metadata
                                })
                        
                        # Emit progress update
                        socketio.emit('batch_progress', {
                            'completed': min(i + chunk_size, len(requests)),
                            'total': len(requests),
                            'percentage': min(100, ((i + chunk_size) / len(requests)) * 100)
                        }, room=client_id)
                    
                    return results
                
                results = loop.run_until_complete(process_batch())
                
                # Emit batch completion
                socketio.emit('batch_completed', {
                    'total_processed': len(results),
                    'successful': len([r for r in results if 'error' not in r]),
                    'failed': len([r for r in results if 'error' in r]),
                    'results': results,
                    'timestamp': datetime.utcnow().isoformat()
                }, room=client_id)
                
            finally:
                loop.close()
                
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        socketio.emit('error', {
            'message': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }, room=client_id)

@socketio.on('get_live_metrics')
def handle_get_live_metrics():
    """Get live performance metrics"""
    client_id = request.sid
    
    try:
        framework = get_framework()
        
        # Get current metrics
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            metrics = loop.run_until_complete(
                framework.get_performance_metrics(time_window_hours=1)
            )
        finally:
            loop.close()
        
        emit('live_metrics', {
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting live metrics: {e}")
        emit('error', {'message': str(e)})

@socketio.on('subscribe_metrics')
def handle_subscribe_metrics(data):
    """Subscribe to live metrics updates"""
    client_id = request.sid
    interval = data.get('interval', 5)  # Default 5 seconds
    
    # Join metrics room
    join_room('metrics_subscribers')
    
    if client_id in active_sessions:
        active_sessions[client_id]['rooms'].add('metrics_subscribers')
    
    emit('subscribed_to_metrics', {'interval': interval})

# Background task to emit periodic metrics updates
def emit_periodic_metrics():
    """Emit metrics updates to all subscribers"""
    while True:
        try:
            with app.app_context():
                framework = get_framework()
                
                # Get metrics
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    metrics = loop.run_until_complete(
                        framework.get_performance_metrics(time_window_hours=1)
                    )
                finally:
                    loop.close()
                
                # Emit to all subscribers
                socketio.emit('metrics_update', {
                    'metrics': metrics,
                    'timestamp': datetime.utcnow().isoformat()
                }, room='metrics_subscribers')
                
        except Exception as e:
            logger.error(f"Error emitting periodic metrics: {e}")
        
        # Wait before next update
        socketio.sleep(5)

# Start background metrics task when module loads
socketio.start_background_task(emit_periodic_metrics)