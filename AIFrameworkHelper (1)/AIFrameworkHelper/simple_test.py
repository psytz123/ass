"""
Simple test to validate the AI Orchestration Framework
"""
import asyncio
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app
from config import Config
from core.framework import AIOrchestrationFramework

async def test_basic_functionality():
    """Test basic framework functionality"""
    print("Testing AI Orchestration Framework...")
    
    # Create framework with default config
    config = Config()
    framework = AIOrchestrationFramework(config)
    
    print(f"Framework initialized with {len(framework.model_connectors)} connectors")
    
    # Test basic request
    start_time = time.time()
    
    try:
        result = await framework.process_request(
            prompt="Hello, please respond with a simple greeting",
            task_type="general",
            user_id="test_user"
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        print(f"✓ Request processed successfully in {processing_time:.2f}ms")
        print(f"✓ Response: {result.final_response.content}")
        print(f"✓ Providers used: {[p.value for p in result.providers_used]}")
        print(f"✓ Complexity score: {result.complexity_score}")
        
        if result.consensus_result:
            print(f"✓ Consensus confidence: {result.consensus_result.confidence}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_performance_metrics():
    """Test performance metrics collection"""
    print("\nTesting performance metrics...")
    
    config = Config()
    framework = AIOrchestrationFramework(config)
    
    # Make a few requests
    for i in range(3):
        await framework.process_request(
            prompt=f"Test request {i+1}",
            task_type="general",
            user_id="metrics_user"
        )
    
    # Get metrics
    metrics = await framework.get_performance_metrics(time_window_hours=1)
    
    if metrics and metrics.get('total_requests', 0) > 0:
        print(f"✓ Performance metrics collected: {metrics['total_requests']} requests")
        print(f"✓ Average latency: {metrics.get('avg_latency_ms', 0):.2f}ms")
        print(f"✓ Success rate: {metrics.get('success_rate', 0):.2f}")
        return True
    else:
        print("✗ Performance metrics not collected")
        return False

async def test_conversation_history():
    """Test conversation history storage"""
    print("\nTesting conversation history...")
    
    config = Config()
    framework = AIOrchestrationFramework(config)
    
    user_id = "history_test_user"
    
    # Make a few requests
    for i in range(2):
        await framework.process_request(
            prompt=f"History test question {i+1}",
            task_type="general",
            user_id=user_id
        )
    
    # Get history
    history = await framework.get_conversation_history(user_id, limit=5)
    
    if len(history) >= 2:
        print(f"✓ Conversation history stored: {len(history)} conversations")
        for i, conv in enumerate(history[:2]):
            print(f"  {i+1}. {conv['prompt'][:50]}...")
        return True
    else:
        print("✗ Conversation history not stored properly")
        return False

async def main():
    """Run all tests"""
    print("=" * 60)
    print("AI ORCHESTRATION FRAMEWORK TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Performance Metrics", test_performance_metrics),
        ("Conversation History", test_conversation_history),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The AI Orchestration Framework is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    asyncio.run(main())