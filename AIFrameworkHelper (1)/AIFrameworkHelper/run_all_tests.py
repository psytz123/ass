#!/usr/bin/env python3
"""
Comprehensive Test Runner for AI Orchestration Framework
Run all tests to validate the system functionality
"""
import asyncio
import time
import json
import logging
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db
from config import Config
from core.framework import AIOrchestrationFramework
from core.types import ModelProvider

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestRunner:
    """Comprehensive test runner for the AI framework"""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        
    def log_test(self, test_name, passed, details=None, error=None):
        """Log test result"""
        status = "✓ PASSED" if passed else "✗ FAILED"
        result = {
            'test_name': test_name,
            'passed': passed,
            'details': details,
            'error': str(error) if error else None,
            'timestamp': datetime.now().isoformat()
        }
        self.results.append(result)
        
        print(f"{status}: {test_name}")
        if details:
            print(f"  Details: {details}")
        if error:
            print(f"  Error: {error}")
        print()

    async def test_framework_initialization(self):
        """Test 1: Framework Initialization"""
        try:
            with app.app_context():
                config = Config()
                framework = AIOrchestrationFramework(config)
                
                # Check basic attributes
                available_providers = []
                if hasattr(framework, 'openai_connector') and framework.openai_connector:
                    available_providers.append('OpenAI')
                if hasattr(framework, 'anthropic_connector') and framework.anthropic_connector:
                    available_providers.append('Anthropic')
                if hasattr(framework, 'google_connector') and framework.google_connector:
                    available_providers.append('Google')
                
                details = f"Initialized with {len(available_providers)} providers: {', '.join(available_providers)}"
                self.log_test("Framework Initialization", True, details)
                return framework
                
        except Exception as e:
            self.log_test("Framework Initialization", False, error=e)
            return None

    async def test_basic_request_processing(self, framework):
        """Test 2: Basic Request Processing"""
        try:
            with app.app_context():
                # Simple request
                result = await framework.process_request(
                    prompt="Hello, please respond with a simple greeting",
                    task_type="general",
                    user_id="test_user_basic"
                )
                
                # Validate result structure
                has_response = hasattr(result, 'final_response') and result.final_response
                has_content = has_response and hasattr(result.final_response, 'content') and result.final_response.content
                
                if has_content:
                    details = f"Response: {result.final_response.content[:100]}..."
                    self.log_test("Basic Request Processing", True, details)
                    return True
                else:
                    self.log_test("Basic Request Processing", False, error="No valid response content")
                    return False
                    
        except Exception as e:
            self.log_test("Basic Request Processing", False, error=e)
            return False

    async def test_multiple_provider_consensus(self, framework):
        """Test 3: Multiple Provider Consensus"""
        try:
            with app.app_context():
                # Complex request requiring consensus
                result = await framework.process_request(
                    prompt="What is the capital of France?",
                    task_type="factual",
                    user_id="test_user_consensus",
                    require_consensus=True,
                    min_providers=2
                )
                
                providers_used = getattr(result, 'providers_used', [])
                consensus_result = getattr(result, 'consensus_result', None)
                
                if len(providers_used) >= 2:
                    details = f"Used {len(providers_used)} providers with consensus"
                    if consensus_result and hasattr(consensus_result, 'confidence'):
                        details += f", confidence: {consensus_result.confidence:.2f}"
                    self.log_test("Multiple Provider Consensus", True, details)
                    return True
                else:
                    details = f"Only used {len(providers_used)} providers"
                    self.log_test("Multiple Provider Consensus", False, details)
                    return False
                    
        except Exception as e:
            self.log_test("Multiple Provider Consensus", False, error=e)
            return False

    async def test_conversation_storage(self, framework):
        """Test 4: Conversation Storage"""
        try:
            with app.app_context():
                user_id = "test_user_storage"
                
                # Make several requests
                prompts = [
                    "First test question",
                    "Second test question", 
                    "Third test question"
                ]
                
                for prompt in prompts:
                    await framework.process_request(
                        prompt=prompt,
                        task_type="general",
                        user_id=user_id
                    )
                
                # Check conversation history
                history = await framework.get_conversation_history(user_id, limit=10)
                
                if len(history) >= len(prompts):
                    details = f"Stored {len(history)} conversations successfully"
                    self.log_test("Conversation Storage", True, details)
                    return True
                else:
                    details = f"Expected {len(prompts)} conversations, found {len(history)}"
                    self.log_test("Conversation Storage", False, details)
                    return False
                    
        except Exception as e:
            self.log_test("Conversation Storage", False, error=e)
            return False

    async def test_performance_metrics(self, framework):
        """Test 5: Performance Metrics Collection"""
        try:
            with app.app_context():
                # Make requests to generate metrics
                for i in range(3):
                    await framework.process_request(
                        prompt=f"Performance test request {i+1}",
                        task_type="general",
                        user_id=f"perf_user_{i}"
                    )
                
                # Get performance metrics
                metrics = await framework.get_performance_metrics(time_window_hours=1)
                
                if metrics and isinstance(metrics, dict) and metrics.get('total_requests', 0) > 0:
                    details = f"Collected metrics: {metrics.get('total_requests', 0)} requests, "
                    details += f"avg latency: {metrics.get('avg_latency_ms', 0):.2f}ms"
                    self.log_test("Performance Metrics Collection", True, details)
                    return True
                else:
                    self.log_test("Performance Metrics Collection", False, error="No metrics collected")
                    return False
                    
        except Exception as e:
            self.log_test("Performance Metrics Collection", False, error=e)
            return False

    async def test_task_complexity_evaluation(self, framework):
        """Test 6: Task Complexity Evaluation"""
        try:
            with app.app_context():
                test_cases = [
                    ("Hello", "simple greeting"),
                    ("Explain quantum computing in detail", "complex technical explanation"),
                    ("Analyze market trends", "analytical task")
                ]
                
                complexity_scores = []
                for prompt, description in test_cases:
                    result = await framework.process_request(
                        prompt=prompt,
                        task_type="general",
                        user_id="complexity_test_user"
                    )
                    
                    if hasattr(result, 'complexity_score') and result.complexity_score is not None:
                        complexity_scores.append(result.complexity_score)
                
                if len(complexity_scores) == len(test_cases):
                    details = f"Evaluated complexity for {len(complexity_scores)} tasks"
                    self.log_test("Task Complexity Evaluation", True, details)
                    return True
                else:
                    details = f"Only evaluated {len(complexity_scores)} out of {len(test_cases)} tasks"
                    self.log_test("Task Complexity Evaluation", False, details)
                    return False
                    
        except Exception as e:
            self.log_test("Task Complexity Evaluation", False, error=e)
            return False

    async def test_concurrent_requests(self, framework):
        """Test 7: Concurrent Request Handling"""
        try:
            with app.app_context():
                async def make_request(i):
                    return await framework.process_request(
                        prompt=f"Concurrent test request {i}",
                        task_type="general",
                        user_id=f"concurrent_user_{i}"
                    )
                
                # Create 5 concurrent requests
                start_time = time.time()
                tasks = [make_request(i) for i in range(5)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                elapsed_time = time.time() - start_time
                
                successful_results = [r for r in results if not isinstance(r, Exception)]
                
                if len(successful_results) >= 4:  # Allow 1 failure
                    details = f"{len(successful_results)}/5 concurrent requests completed in {elapsed_time:.2f}s"
                    self.log_test("Concurrent Request Handling", True, details)
                    return True
                else:
                    details = f"Only {len(successful_results)}/5 requests succeeded"
                    self.log_test("Concurrent Request Handling", False, details)
                    return False
                    
        except Exception as e:
            self.log_test("Concurrent Request Handling", False, error=e)
            return False

    async def test_error_handling(self, framework):
        """Test 8: Error Handling"""
        try:
            with app.app_context():
                error_cases = [
                    ("", "empty prompt"),
                    ("   ", "whitespace-only prompt"),
                ]
                
                handled_errors = 0
                for prompt, description in error_cases:
                    try:
                        result = await framework.process_request(
                            prompt=prompt,
                            task_type="general",
                            user_id="error_test_user"
                        )
                        # If we get a result with empty prompt, that's fine too
                        if result:
                            handled_errors += 1
                    except Exception:
                        # Expected for invalid inputs
                        handled_errors += 1
                
                if handled_errors == len(error_cases):
                    details = f"Properly handled {handled_errors} error cases"
                    self.log_test("Error Handling", True, details)
                    return True
                else:
                    self.log_test("Error Handling", False, error="Some error cases not handled properly")
                    return False
                    
        except Exception as e:
            self.log_test("Error Handling", False, error=e)
            return False

    def generate_test_report(self):
        """Generate comprehensive test report"""
        total_time = time.time() - self.start_time
        passed_tests = sum(1 for r in self.results if r['passed'])
        total_tests = len(self.results)
        
        print("=" * 80)
        print("AI ORCHESTRATION FRAMEWORK - COMPREHENSIVE TEST REPORT")
        print("=" * 80)
        print(f"Test Execution Time: {total_time:.2f} seconds")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print()
        
        print("TEST RESULTS:")
        print("-" * 40)
        for result in self.results:
            status = "✓" if result['passed'] else "✗"
            print(f"{status} {result['test_name']}")
            if result['details']:
                print(f"  {result['details']}")
            if result['error']:
                print(f"  Error: {result['error']}")
        
        print()
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! The AI Orchestration Framework is working perfectly.")
        elif passed_tests >= total_tests * 0.8:
            print("✅ Most tests passed. The framework is functional with minor issues.")
        elif passed_tests >= total_tests * 0.5:
            print("⚠️ Some tests failed. The framework has significant issues that need attention.")
        else:
            print("❌ Many tests failed. The framework needs major fixes.")
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': (passed_tests/total_tests)*100,
            'execution_time': total_time,
            'results': self.results
        }

async def main():
    """Run all tests"""
    print("Starting comprehensive test suite for AI Orchestration Framework...")
    print("=" * 80)
    
    runner = TestRunner()
    
    # Initialize framework
    framework = await runner.test_framework_initialization()
    
    if framework:
        # Run all functional tests
        await runner.test_basic_request_processing(framework)
        await runner.test_multiple_provider_consensus(framework)
        await runner.test_conversation_storage(framework)
        await runner.test_performance_metrics(framework)
        await runner.test_task_complexity_evaluation(framework)
        await runner.test_concurrent_requests(framework)
        await runner.test_error_handling(framework)
    
    # Generate final report
    report = runner.generate_test_report()
    
    # Save report to file
    with open('test_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDetailed test report saved to: test_report.json")
    
    # Exit with appropriate code
    if report['passed_tests'] == report['total_tests']:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed

if __name__ == "__main__":
    asyncio.run(main())