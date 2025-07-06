"""
API endpoints for ML-based routing management
"""

from flask import Blueprint, jsonify, request, render_template_string
import asyncio
import logging

from core.ml_routing.ml_routing_manager import MLRoutingManager
from core.types import TaskComplexity

logger = logging.getLogger(__name__)

ml_routing_bp = Blueprint('ml_routing', __name__, url_prefix='/api')
ml_manager = None

def get_ml_manager():
    """Get or create ML routing manager singleton"""
    global ml_manager
    if ml_manager is None:
        ml_manager = MLRoutingManager()
        # Initialize the ML system
        try:
            asyncio.run(ml_manager.initialize())
        except Exception as e:
            logger.error(f"Failed to initialize ML routing: {e}")
    return ml_manager

@ml_routing_bp.route('/ml-routing')
def ml_routing_dashboard():
    """ML Routing management dashboard"""
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ML Routing Management - AI Orchestration</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #1a1a2e; color: #eee; }
            .card { background-color: #16213e; border: none; margin-bottom: 20px; }
            .card-header { background-color: #0f3460; border-bottom: 1px solid #e94560; }
            .btn-primary { background-color: #e94560; border-color: #e94560; }
            .btn-primary:hover { background-color: #c23850; border-color: #c23850; }
            .badge { font-size: 0.9rem; }
            .feature-box { padding: 15px; border-radius: 5px; background-color: #0f3460; margin: 10px 0; }
            .nav-link.active { background-color: #e94560 !important; border-color: #e94560 !important; }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-dark" style="background-color: #0f3460;">
            <div class="container">
                <a class="navbar-brand" href="/">AI Orchestration Framework</a>
                <div class="d-flex">
                    <a href="/dashboard" class="btn btn-outline-light me-2">Dashboard</a>
                    <a href="/test" class="btn btn-outline-light me-2">Test</a>
                    <a href="/advanced" class="btn btn-outline-light">Advanced</a>
                </div>
            </div>
        </nav>

        <div class="container mt-4">
            <h1>ML-Based Intelligent Routing</h1>
            <p class="lead">Machine Learning powered task complexity evaluation and routing</p>

            <!-- Status Card -->
            <div class="card">
                <div class="card-header">
                    <h4>System Status</h4>
                </div>
                <div class="card-body" id="status-section">
                    <div class="text-center">
                        <div class="spinner-border" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Training Data Card -->
            <div class="card">
                <div class="card-header">
                    <h4>Training Data Collection</h4>
                </div>
                <div class="card-body">
                    <form id="add-prompt-form">
                        <div class="mb-3">
                            <label for="prompt" class="form-label">Prompt</label>
                            <textarea class="form-control" id="prompt" rows="3" required></textarea>
                        </div>
                        <div class="row">
                            <div class="col-md-6">
                                <label for="task-type" class="form-label">Task Type</label>
                                <select class="form-select" id="task-type" required>
                                    <option value="code_generation">Code Generation</option>
                                    <option value="code_optimization">Code Optimization</option>
                                    <option value="text_generation">Text Generation</option>
                                    <option value="analysis">Analysis</option>
                                    <option value="question_answering">Question Answering</option>
                                    <option value="other">Other</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label for="complexity" class="form-label">Complexity</label>
                                <select class="form-select" id="complexity" required>
                                    <option value="simple">Simple</option>
                                    <option value="medium">Medium</option>
                                    <option value="complex">Complex</option>
                                </select>
                            </div>
                        </div>
                        <button type="submit" class="btn btn-primary mt-3">Add Training Sample</button>
                    </form>
                    <div id="collection-result" class="mt-3"></div>
                </div>
            </div>

            <!-- Model Management Card -->
            <div class="card">
                <div class="card-header">
                    <h4>Model Management</h4>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">
                            <button class="btn btn-primary w-100" onclick="retrainModel()">
                                Retrain Model
                            </button>
                        </div>
                        <div class="col-md-6">
                            <button class="btn btn-secondary w-100" onclick="collectFromDatabase()">
                                Collect Data from Database
                            </button>
                        </div>
                    </div>
                    <div id="model-result" class="mt-3"></div>
                </div>
            </div>

            <!-- Test Model Card -->
            <div class="card">
                <div class="card-header">
                    <h4>Test ML Routing</h4>
                </div>
                <div class="card-body">
                    <form id="test-form">
                        <div class="mb-3">
                            <label for="test-prompt" class="form-label">Test Prompt</label>
                            <textarea class="form-control" id="test-prompt" rows="2" required></textarea>
                        </div>
                        <div class="mb-3">
                            <label for="test-task-type" class="form-label">Task Type</label>
                            <select class="form-select" id="test-task-type" required>
                                <option value="code_generation">Code Generation</option>
                                <option value="code_optimization">Code Optimization</option>
                                <option value="text_generation">Text Generation</option>
                                <option value="analysis">Analysis</option>
                                <option value="question_answering">Question Answering</option>
                            </select>
                        </div>
                        <button type="submit" class="btn btn-primary">Evaluate Complexity</button>
                    </form>
                    <div id="test-result" class="mt-3"></div>
                </div>
            </div>
        </div>

        <script>
            // Load status on page load
            window.onload = function() {
                loadStatus();
            };

            async function loadStatus() {
                try {
                    const response = await fetch('/api/ml-routing/status');
                    const data = await response.json();
                    
                    const statusHtml = `
                        <div class="row">
                            <div class="col-md-6">
                                <div class="feature-box">
                                    <h5>Model Status</h5>
                                    <p>Model Exists: <span class="badge ${data.model_exists ? 'bg-success' : 'bg-warning'}">${data.model_exists ? 'Yes' : 'No'}</span></p>
                                    <p>Model Path: <code>${data.model_path}</code></p>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="feature-box">
                                    <h5>Dataset Statistics</h5>
                                    <p>Total Samples: <strong>${data.dataset_stats.total}</strong></p>
                                    <p>Manual Labels: <strong>${data.dataset_stats.manual_labels || 0}</strong></p>
                                    <p>Avg Prompt Length: <strong>${Math.round(data.dataset_stats.avg_prompt_length || 0)}</strong> chars</p>
                                </div>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-md-6">
                                <div class="feature-box">
                                    <h5>By Task Type</h5>
                                    ${Object.entries(data.dataset_stats.by_task_type || {}).map(([type, count]) => 
                                        `<p>${type}: <strong>${count}</strong></p>`
                                    ).join('')}
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="feature-box">
                                    <h5>By Complexity</h5>
                                    ${Object.entries(data.dataset_stats.by_complexity || {}).map(([complexity, count]) => 
                                        `<p>${complexity}: <strong>${count}</strong></p>`
                                    ).join('')}
                                </div>
                            </div>
                        </div>
                    `;
                    
                    document.getElementById('status-section').innerHTML = statusHtml;
                } catch (error) {
                    document.getElementById('status-section').innerHTML = 
                        '<div class="alert alert-danger">Error loading status</div>';
                }
            }

            // Add training sample
            document.getElementById('add-prompt-form').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const prompt = document.getElementById('prompt').value;
                const taskType = document.getElementById('task-type').value;
                const complexity = document.getElementById('complexity').value;
                
                try {
                    const response = await fetch('/api/ml-routing/add-sample', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({prompt, task_type: taskType, complexity})
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        document.getElementById('collection-result').innerHTML = 
                            '<div class="alert alert-success">Sample added successfully!</div>';
                        document.getElementById('add-prompt-form').reset();
                        loadStatus(); // Refresh status
                    } else {
                        document.getElementById('collection-result').innerHTML = 
                            `<div class="alert alert-warning">${result.message || 'Failed to add sample'}</div>`;
                    }
                } catch (error) {
                    document.getElementById('collection-result').innerHTML = 
                        '<div class="alert alert-danger">Error adding sample</div>';
                }
            });

            // Test model
            document.getElementById('test-form').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const prompt = document.getElementById('test-prompt').value;
                const taskType = document.getElementById('test-task-type').value;
                
                try {
                    const response = await fetch('/api/ml-routing/evaluate', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({prompt, task_type: taskType})
                    });
                    
                    const result = await response.json();
                    
                    const resultHtml = `
                        <div class="feature-box">
                            <h5>Evaluation Result</h5>
                            <p>Complexity: <span class="badge bg-primary">${result.complexity}</span></p>
                            <p>Confidence: <strong>${(result.confidence * 100).toFixed(1)}%</strong></p>
                            <p>Evaluator: <strong>${result.evaluator_type}</strong></p>
                        </div>
                    `;
                    
                    document.getElementById('test-result').innerHTML = resultHtml;
                } catch (error) {
                    document.getElementById('test-result').innerHTML = 
                        '<div class="alert alert-danger">Error evaluating prompt</div>';
                }
            });

            async function retrainModel() {
                document.getElementById('model-result').innerHTML = 
                    '<div class="text-center"><div class="spinner-border" role="status"></div> Training model...</div>';
                
                try {
                    const response = await fetch('/api/ml-routing/retrain', {method: 'POST'});
                    const result = await response.json();
                    
                    if (result.success) {
                        const trainingResults = result.training_results;
                        const resultHtml = `
                            <div class="alert alert-success">
                                <h5>Model Retrained Successfully!</h5>
                                <p>Test F1 Score: <strong>${trainingResults.test_f1_weighted.toFixed(3)}</strong></p>
                                <p>CV F1 Score: <strong>${trainingResults.cv_f1_mean.toFixed(3)} ± ${trainingResults.cv_f1_std.toFixed(3)}</strong></p>
                                <p>Test Accuracy: <strong>${(trainingResults.test_accuracy * 100).toFixed(1)}%</strong></p>
                            </div>
                        `;
                        document.getElementById('model-result').innerHTML = resultHtml;
                        loadStatus();
                    } else {
                        document.getElementById('model-result').innerHTML = 
                            `<div class="alert alert-danger">Error: ${result.error}</div>`;
                    }
                } catch (error) {
                    document.getElementById('model-result').innerHTML = 
                        '<div class="alert alert-danger">Error retraining model</div>';
                }
            }

            async function collectFromDatabase() {
                try {
                    const response = await fetch('/api/ml-routing/collect-data', {method: 'POST'});
                    const result = await response.json();
                    
                    document.getElementById('model-result').innerHTML = 
                        `<div class="alert alert-info">Collected ${result.new_samples} new samples from database</div>`;
                    loadStatus();
                } catch (error) {
                    document.getElementById('model-result').innerHTML = 
                        '<div class="alert alert-danger">Error collecting data</div>';
                }
            }
        </script>
    </body>
    </html>
    """)

@ml_routing_bp.route('/api/ml-routing/status')
def ml_routing_status():
    """Get ML routing system status"""
    manager = get_ml_manager()
    stats = manager.get_stats()
    return jsonify(stats)

@ml_routing_bp.route('/api/ml-routing/add-sample', methods=['POST'])
def add_training_sample():
    """Add a manual training sample"""
    data = request.json
    manager = get_ml_manager()
    
    try:
        complexity = TaskComplexity(data['complexity'])
        success = manager.data_collector.add_manual_prompt(
            prompt=data['prompt'],
            task_type=data['task_type'],
            complexity=complexity
        )
        
        return jsonify({
            'success': success,
            'message': 'Sample added' if success else 'Duplicate prompt'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@ml_routing_bp.route('/api/ml-routing/evaluate', methods=['POST'])
def evaluate_prompt():
    """Evaluate a prompt using ML routing"""
    data = request.json
    manager = get_ml_manager()
    
    try:
        # Use the evaluator directly
        if hasattr(manager.evaluator, 'evaluate_with_confidence'):
            complexity, confidence = asyncio.run(
                manager.evaluator.evaluate_with_confidence(
                    data['prompt'], 
                    data['task_type']
                )
            )
        else:
            complexity = asyncio.run(
                manager.evaluator.evaluate(
                    data['prompt'], 
                    data['task_type']
                )
            )
            confidence = 0.5
        
        evaluator_type = "ML" if manager.evaluator.model else "Simple"
        
        return jsonify({
            'complexity': complexity.value,
            'confidence': confidence,
            'evaluator_type': evaluator_type
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@ml_routing_bp.route('/api/ml-routing/retrain', methods=['POST'])
def retrain_model():
    """Retrain the ML routing model"""
    manager = get_ml_manager()
    
    try:
        result = asyncio.run(manager.retrain_model())
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@ml_routing_bp.route('/api/ml-routing/collect-data', methods=['POST'])
def collect_data():
    """Collect training data from database"""
    manager = get_ml_manager()
    
    try:
        new_samples = manager.data_collector.collect_from_database(limit=500)
        return jsonify({'success': True, 'new_samples': new_samples})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500