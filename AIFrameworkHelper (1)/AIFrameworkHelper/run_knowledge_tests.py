#!/usr/bin/env python
"""
Run Knowledge Integration Tests
===============================
This script runs the comprehensive knowledge validation tests for the AI Orchestration Framework.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import test components
from tests.test_ai_knowledge_integration import (
    TestAIAssistantIntegration,
    TestKnowledgeComparison,
    TestSpecializedKnowledge,
    TestKnowledgePerformance,
    TestKnowledgeIntegrationSuite,
    KnowledgeValidator,
    AIOrchestrationFramework
)

class KnowledgeTestRunner:
    """Run and report on knowledge integration tests"""
    
    def __init__(self):
        self.results = []
        self.framework = None
        self.validator = None
        
    async def setup(self):
        """Initialize test framework"""
        print("Initializing AI Orchestration Framework...")
        self.framework = AIOrchestrationFramework()
        self.validator = KnowledgeValidator()
        print("Framework initialized successfully!\n")
        
    async def run_basic_integration_tests(self):
        """Run basic integration tests"""
        print("="*60)
        print("RUNNING BASIC INTEGRATION TESTS")
        print("="*60)
        
        test_suite = TestAIAssistantIntegration()
        test_suite.knowledge_framework = self.framework
        test_suite.knowledge_validator = self.validator
        
        try:
            # Test 1: Basic Model Integration
            print("\n1. Testing Basic Model Integration...")
            await test_suite.test_basic_model_integration(self.framework)
            self.results.append(("Basic Model Integration", "PASSED", "All providers integrated successfully"))
        except Exception as e:
            self.results.append(("Basic Model Integration", "FAILED", str(e)))
            
        try:
            # Test 2: Response Consistency
            print("\n2. Testing Response Consistency...")
            await test_suite.test_response_consistency_across_models(self.framework)
            self.results.append(("Response Consistency", "PASSED", "Consistent responses across models"))
        except Exception as e:
            self.results.append(("Response Consistency", "FAILED", str(e)))
            
        try:
            # Test 3: Domain Knowledge
            print("\n3. Testing Domain-Specific Knowledge...")
            await test_suite.test_domain_specific_knowledge(self.framework, self.validator)
            self.results.append(("Domain Knowledge", "PASSED", "Good performance across domains"))
        except Exception as e:
            self.results.append(("Domain Knowledge", "FAILED", str(e)))
            
    async def run_specialized_tests(self):
        """Run specialized knowledge tests"""
        print("\n" + "="*60)
        print("RUNNING SPECIALIZED KNOWLEDGE TESTS")
        print("="*60)
        
        test_suite = TestSpecializedKnowledge()
        
        try:
            # Test 1: Technical Accuracy
            print("\n1. Testing Technical Accuracy...")
            await test_suite.test_technical_accuracy(self.framework)
            self.results.append(("Technical Accuracy", "PASSED", "High technical accuracy"))
        except Exception as e:
            self.results.append(("Technical Accuracy", "FAILED", str(e)))
            
        try:
            # Test 2: Interdisciplinary Knowledge
            print("\n2. Testing Interdisciplinary Knowledge...")
            await test_suite.test_interdisciplinary_knowledge(self.framework)
            self.results.append(("Interdisciplinary Knowledge", "PASSED", "Good cross-domain connections"))
        except Exception as e:
            self.results.append(("Interdisciplinary Knowledge", "FAILED", str(e)))
            
    async def run_performance_tests(self):
        """Run performance tests"""
        print("\n" + "="*60)
        print("RUNNING PERFORMANCE TESTS")
        print("="*60)
        
        test_suite = TestKnowledgePerformance()
        
        try:
            # Test 1: Query Latency
            print("\n1. Testing Query Latency...")
            await test_suite.test_knowledge_query_latency(self.framework)
            self.results.append(("Query Latency", "PASSED", "Acceptable response times"))
        except Exception as e:
            self.results.append(("Query Latency", "FAILED", str(e)))
            
        try:
            # Test 2: Concurrent Queries
            print("\n2. Testing Concurrent Query Handling...")
            await test_suite.test_concurrent_knowledge_queries(self.framework)
            self.results.append(("Concurrent Queries", "PASSED", "Good concurrent handling"))
        except Exception as e:
            self.results.append(("Concurrent Queries", "FAILED", str(e)))
            
    def generate_report(self):
        """Generate final test report"""
        print("\n" + "="*80)
        print("KNOWLEDGE INTEGRATION TEST REPORT")
        print("="*80)
        print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        passed = sum(1 for _, status, _ in self.results if status == "PASSED")
        total = len(self.results)
        
        print(f"\nOverall Result: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
        print("\nDetailed Results:")
        print("-"*80)
        
        for test_name, status, details in self.results:
            status_emoji = "✓" if status == "PASSED" else "✗"
            print(f"{status_emoji} {test_name}: {status}")
            if status == "FAILED" or details != "":
                print(f"   Details: {details}")
        
        print("-"*80)
        
        if passed == total:
            print("\n🎉 All tests passed! The AI Orchestration Framework is working correctly.")
        elif passed >= total * 0.7:
            print(f"\n⚠️  Most tests passed ({passed}/{total}), but some issues need attention.")
        else:
            print(f"\n❌ Multiple test failures ({total-passed}/{total}). Please review the errors.")
            
async def main():
    """Main test execution"""
    runner = KnowledgeTestRunner()
    
    try:
        # Setup
        await runner.setup()
        
        # Run test suites
        await runner.run_basic_integration_tests()
        await runner.run_specialized_tests()
        await runner.run_performance_tests()
        
    except Exception as e:
        print(f"\n❌ Critical error during test execution: {e}")
        runner.results.append(("Test Execution", "FAILED", str(e)))
    
    finally:
        # Generate report
        runner.generate_report()

if __name__ == "__main__":
    print("AI Orchestration Framework - Knowledge Integration Tests")
    print("="*60)
    asyncio.run(main())