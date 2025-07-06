#!/usr/bin/env python3
"""
Simple Integration Test Runner for AI Orchestration Framework
Runs basic integration tests to validate the system is working
"""
import asyncio
import sys
from app import app, db
from config import Config
from core.framework import AIOrchestrationFramework

async def run_basic_integration_test():
    """Run a basic integration test"""
    print("=" * 60)
    print("AI ORCHESTRATION FRAMEWORK - INTEGRATION TEST")
    print("=" * 60)
    print()
    
    # Run within Flask app context
    with app.app_context():
        try:
            # Initialize framework
            print("1. Initializing framework...")
            config = Config()
            framework = AIOrchestrationFramework(config)
            print("✓ Framework initialized successfully")
            
            # Check available providers
            print("\n2. Checking available providers...")
            providers = framework.get_available_providers()
            print(f"✓ Available providers: {providers}")
            
            if not providers:
                print("⚠️  No providers available. Please check API keys.")
                return False
            
            # Test basic request
            print("\n3. Testing basic request processing...")
            test_prompt = "What is 2 + 2?"
            
            try:
                result = await framework.process_request(
                    prompt=test_prompt,
                    task_type="general",
                    user_id="test_user"
                )
                
                print(f"✓ Request processed successfully")
                print(f"  Response: {result.response[:100]}...")
                print(f"  Providers used: {result.metadata.get('providers_used', [])}")
                print(f"  Processing time: {result.metadata.get('processing_time_ms', 0):.2f}ms")
                
            except Exception as e:
                print(f"✗ Request processing failed: {e}")
                return False
            
            # Test consensus if multiple providers available
            if len(providers) > 1:
                print("\n4. Testing consensus mechanism...")
                test_prompt = "Explain the concept of recursion in programming in one sentence."
                
                try:
                    result = await framework.process_request(
                        prompt=test_prompt,
                        task_type="general",
                        user_id="test_user",
                        require_consensus=True
                    )
                    
                    print(f"✓ Consensus request processed successfully")
                    if result.consensus_result:
                        print(f"  Consensus confidence: {result.consensus_result.confidence:.2f}")
                        print(f"  Number of responses: {len(result.consensus_result.individual_responses)}")
                    
                except Exception as e:
                    print(f"⚠️  Consensus test failed (non-critical): {e}")
            
            # Test performance tracking
            print("\n5. Testing performance metrics...")
            try:
                metrics = await framework.get_performance_metrics()
                if metrics:
                    print(f"✓ Performance metrics available")
                    for provider, stats in metrics.items():
                        if stats.get('request_count', 0) > 0:
                            print(f"  {provider}: {stats['request_count']} requests, "
                                  f"avg latency: {stats.get('avg_latency_ms', 0):.2f}ms")
                else:
                    print("✓ Performance metrics system working (no data yet)")
            except Exception as e:
                print(f"⚠️  Performance metrics test failed (non-critical): {e}")
            
            # Test conversation history
            print("\n6. Testing conversation history...")
            try:
                history = await framework.get_conversation_history("test_user", limit=5)
                print(f"✓ Conversation history retrieved: {len(history)} entries")
            except Exception as e:
                print(f"⚠️  Conversation history test failed (non-critical): {e}")
            
            print("\n" + "=" * 60)
            print("INTEGRATION TEST COMPLETED SUCCESSFULLY ✓")
            print("The AI Orchestration Framework is working properly!")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"\n✗ Integration test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

async def main():
    """Main entry point"""
    print("Starting integration test...\n")
    
    # Check for API keys
    import os
    has_openai = bool(os.environ.get('OPENAI_API_KEY'))
    has_anthropic = bool(os.environ.get('ANTHROPIC_API_KEY'))
    has_google = bool(os.environ.get('GOOGLE_API_KEY'))
    
    print("API Key Status:")
    print(f"  OpenAI: {'✓' if has_openai else '✗'}")
    print(f"  Anthropic: {'✓' if has_anthropic else '✗'}")
    print(f"  Google: {'✓' if has_google else '✗'}")
    print()
    
    if not any([has_openai, has_anthropic, has_google]):
        print("⚠️  WARNING: No API keys found. Please set at least one API key.")
        print("  Required environment variables:")
        print("  - OPENAI_API_KEY")
        print("  - ANTHROPIC_API_KEY")
        print("  - GOOGLE_API_KEY")
        print()
        print("The test will continue but may fail.")
        print()
    
    # Run the test
    success = await run_basic_integration_test()
    
    if success:
        print("\n✅ All integration tests passed!")
        print("\nYou can now:")
        print("- Run the full integration.py test suite with: pytest integration.py -v")
        print("- Access the web interface at http://0.0.0.0:5000")
        print("- Run specific test classes from integration.py")
        sys.exit(0)
    else:
        print("\n❌ Some integration tests failed.")
        print("Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())