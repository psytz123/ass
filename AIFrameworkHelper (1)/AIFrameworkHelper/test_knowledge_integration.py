#!/usr/bin/env python3
"""
Knowledge Integration Test Runner
================================
Tests the AI Orchestration Framework's knowledge capabilities across different domains
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Any

# Import framework components
from app import app, db
from config import Config
from core.framework import AIOrchestrationFramework
from core.types import ModelProvider

class KnowledgeTestRunner:
    """Run knowledge validation tests for the AI framework"""
    
    def __init__(self):
        self.results = []
        self.framework = None
        
    async def initialize(self):
        """Initialize the test framework"""
        with app.app_context():
            config = Config()
            self.framework = AIOrchestrationFramework(config)
            await self.framework.initialize()
    
    def log_test(self, test_name: str, passed: bool, details: str = None, score: float = None):
        """Log test results"""
        result = {
            "test": test_name,
            "passed": passed,
            "details": details,
            "score": score,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)
        
        # Print result
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
        if details:
            print(f"  Details: {details}")
        if score is not None:
            print(f"  Score: {score:.2f}")
        print()
    
    async def test_basic_knowledge(self):
        """Test basic factual knowledge"""
        print("\n" + "="*60)
        print("TESTING BASIC KNOWLEDGE")
        print("="*60)
        
        test_cases = [
            {
                "question": "What is the capital of France?",
                "expected": ["paris"],
                "domain": "geography"
            },
            {
                "question": "Who wrote Romeo and Juliet?",
                "expected": ["shakespeare"],
                "domain": "literature"
            },
            {
                "question": "What is 2 + 2?",
                "expected": ["4", "four"],
                "domain": "mathematics"
            },
            {
                "question": "What is the chemical symbol for water?",
                "expected": ["h2o", "h₂o"],
                "domain": "science"
            }
        ]
        
        scores = []
        
        for test in test_cases:
            try:
                result = await self.framework.process_request(
                    prompt=test["question"],
                    task_type="factual_query",
                    user_id="knowledge_test"
                )
                
                response = result["response"].lower()
                correct = any(expected in response for expected in test["expected"])
                score = 1.0 if correct else 0.0
                scores.append(score)
                
                self.log_test(
                    f"Basic Knowledge - {test['domain']}",
                    correct,
                    f"Q: {test['question']}",
                    score
                )
                
            except Exception as e:
                self.log_test(
                    f"Basic Knowledge - {test['domain']}",
                    False,
                    f"Error: {str(e)}",
                    0.0
                )
                scores.append(0.0)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"\nBasic Knowledge Average Score: {avg_score:.2f}")
        return avg_score
    
    async def test_programming_knowledge(self):
        """Test programming and technical knowledge"""
        print("\n" + "="*60)
        print("TESTING PROGRAMMING KNOWLEDGE")
        print("="*60)
        
        test_cases = [
            {
                "question": "Explain the difference between a list and a tuple in Python",
                "keywords": ["mutable", "immutable", "list", "tuple"],
                "topic": "Python data structures"
            },
            {
                "question": "What is Big O notation and why is it important?",
                "keywords": ["complexity", "algorithm", "efficiency", "performance"],
                "topic": "Algorithm complexity"
            },
            {
                "question": "Write a simple Python function to calculate factorial",
                "keywords": ["def", "factorial", "return", "recursive", "iterative"],
                "topic": "Code generation"
            }
        ]
        
        scores = []
        
        for test in test_cases:
            try:
                result = await self.framework.process_request(
                    prompt=test["question"],
                    task_type="technical_query",
                    user_id="programming_test"
                )
                
                response = result["response"].lower()
                keywords_found = sum(1 for keyword in test["keywords"] if keyword in response)
                score = keywords_found / len(test["keywords"])
                scores.append(score)
                
                self.log_test(
                    f"Programming - {test['topic']}",
                    score >= 0.5,
                    f"Keywords found: {keywords_found}/{len(test['keywords'])}",
                    score
                )
                
            except Exception as e:
                self.log_test(
                    f"Programming - {test['topic']}",
                    False,
                    f"Error: {str(e)}",
                    0.0
                )
                scores.append(0.0)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"\nProgramming Knowledge Average Score: {avg_score:.2f}")
        return avg_score
    
    async def test_reasoning_ability(self):
        """Test logical reasoning and problem-solving"""
        print("\n" + "="*60)
        print("TESTING REASONING ABILITY")
        print("="*60)
        
        test_cases = [
            {
                "question": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                "expected_answer": "5",
                "topic": "Logical reasoning"
            },
            {
                "question": "A farmer has 15 cows. All but 8 die. How many are left?",
                "expected_answer": "8",
                "topic": "Reading comprehension"
            }
        ]
        
        scores = []
        
        for test in test_cases:
            try:
                result = await self.framework.process_request(
                    prompt=test["question"],
                    task_type="reasoning_task",
                    user_id="reasoning_test"
                )
                
                response = result["response"].lower()
                correct = test["expected_answer"] in response
                score = 1.0 if correct else 0.0
                scores.append(score)
                
                self.log_test(
                    f"Reasoning - {test['topic']}",
                    correct,
                    f"Expected '{test['expected_answer']}' in response",
                    score
                )
                
            except Exception as e:
                self.log_test(
                    f"Reasoning - {test['topic']}",
                    False,
                    f"Error: {str(e)}",
                    0.0
                )
                scores.append(0.0)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"\nReasoning Ability Average Score: {avg_score:.2f}")
        return avg_score
    
    async def test_creative_capabilities(self):
        """Test creative and generative abilities"""
        print("\n" + "="*60)
        print("TESTING CREATIVE CAPABILITIES")
        print("="*60)
        
        test_cases = [
            {
                "question": "Write a haiku about artificial intelligence",
                "min_length": 30,
                "topic": "Poetry generation"
            },
            {
                "question": "Generate 3 creative uses for a paperclip besides holding papers",
                "min_length": 50,
                "topic": "Creative thinking"
            }
        ]
        
        scores = []
        
        for test in test_cases:
            try:
                result = await self.framework.process_request(
                    prompt=test["question"],
                    task_type="creative_task",
                    user_id="creative_test"
                )
                
                response = result["response"]
                length_ok = len(response) >= test["min_length"]
                score = 1.0 if length_ok else 0.5
                scores.append(score)
                
                self.log_test(
                    f"Creative - {test['topic']}",
                    length_ok,
                    f"Response length: {len(response)} chars",
                    score
                )
                
            except Exception as e:
                self.log_test(
                    f"Creative - {test['topic']}",
                    False,
                    f"Error: {str(e)}",
                    0.0
                )
                scores.append(0.0)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"\nCreative Capabilities Average Score: {avg_score:.2f}")
        return avg_score
    
    async def test_model_comparison(self):
        """Compare responses across different AI models"""
        print("\n" + "="*60)
        print("TESTING MODEL COMPARISON")
        print("="*60)
        
        test_prompt = "Explain the concept of machine learning in simple terms"
        
        provider_responses = {}
        
        # Test available providers
        for provider in [ModelProvider.OPENAI, ModelProvider.ANTHROPIC]:
            try:
                # Check if provider is available
                connector = self.framework.model_connectors.get(provider)
                if not connector:
                    continue
                
                result = await self.framework.process_request(
                    prompt=test_prompt,
                    task_type="comparison_test",
                    user_id="model_comparison",
                    provider_override=provider
                )
                
                provider_responses[provider.value] = {
                    "response": result["response"],
                    "length": len(result["response"]),
                    "processing_time": result["metadata"].get("processing_time_ms", 0)
                }
                
                self.log_test(
                    f"Model Comparison - {provider.value}",
                    True,
                    f"Response length: {len(result['response'])} chars",
                    1.0
                )
                
            except Exception as e:
                self.log_test(
                    f"Model Comparison - {provider.value}",
                    False,
                    f"Error: {str(e)}",
                    0.0
                )
        
        # Compare responses
        if len(provider_responses) >= 2:
            print("\nModel Comparison Summary:")
            for provider, data in provider_responses.items():
                print(f"  {provider}: {data['length']} chars, {data['processing_time']:.0f}ms")
    
    async def test_consensus_mechanism(self):
        """Test consensus across multiple models"""
        print("\n" + "="*60)
        print("TESTING CONSENSUS MECHANISM")
        print("="*60)
        
        test_cases = [
            "What are the primary colors?",
            "Explain photosynthesis in one sentence"
        ]
        
        for prompt in test_cases:
            try:
                result = await self.framework.process_request(
                    prompt=prompt,
                    task_type="consensus_test",
                    user_id="consensus_test",
                    require_consensus=True
                )
                
                consensus_achieved = result["metadata"].get("consensus_confidence", 0) > 0.7
                providers_used = result["metadata"].get("providers_used", [])
                
                self.log_test(
                    f"Consensus - '{prompt[:30]}...'",
                    consensus_achieved,
                    f"Providers: {len(providers_used)}, Confidence: {result['metadata'].get('consensus_confidence', 0):.2f}",
                    result["metadata"].get("consensus_confidence", 0)
                )
                
            except Exception as e:
                self.log_test(
                    f"Consensus - '{prompt[:30]}...'",
                    False,
                    f"Error: {str(e)}",
                    0.0
                )
    
    def generate_report(self):
        """Generate final test report"""
        print("\n" + "="*80)
        print("KNOWLEDGE INTEGRATION TEST REPORT")
        print("="*80)
        
        # Calculate statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["passed"])
        avg_score = sum(r.get("score", 0) for r in self.results if r.get("score") is not None) / total_tests if total_tests > 0 else 0
        
        print(f"\nTotal Tests Run: {total_tests}")
        print(f"Tests Passed: {passed_tests}/{total_tests} ({(passed_tests/total_tests)*100:.1f}%)")
        print(f"Average Score: {avg_score:.2f}")
        
        # Group by test category
        categories = {}
        for result in self.results:
            category = result["test"].split(" - ")[0]
            if category not in categories:
                categories[category] = {"passed": 0, "total": 0, "scores": []}
            categories[category]["total"] += 1
            if result["passed"]:
                categories[category]["passed"] += 1
            if result.get("score") is not None:
                categories[category]["scores"].append(result["score"])
        
        print("\nResults by Category:")
        for category, data in categories.items():
            pass_rate = (data["passed"] / data["total"]) * 100 if data["total"] > 0 else 0
            avg_cat_score = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
            print(f"  {category}: {data['passed']}/{data['total']} passed ({pass_rate:.1f}%), avg score: {avg_cat_score:.2f}")
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"knowledge_test_report_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "pass_rate": (passed_tests/total_tests)*100 if total_tests > 0 else 0,
                    "average_score": avg_score
                },
                "categories": categories,
                "detailed_results": self.results
            }, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_file}")
        
        # Overall assessment
        print("\n" + "="*80)
        if avg_score >= 0.8:
            print("🎉 EXCELLENT: The AI framework demonstrates strong knowledge capabilities!")
        elif avg_score >= 0.6:
            print("✅ GOOD: The AI framework shows solid knowledge performance.")
        elif avg_score >= 0.4:
            print("⚠️ FAIR: The AI framework has room for improvement.")
        else:
            print("❌ NEEDS IMPROVEMENT: The AI framework requires significant enhancements.")
        print("="*80)

async def main():
    """Run all knowledge integration tests"""
    print("AI ORCHESTRATION FRAMEWORK - KNOWLEDGE INTEGRATION TESTS")
    print("Starting comprehensive knowledge validation...\n")
    
    runner = KnowledgeTestRunner()
    
    try:
        # Initialize framework
        await runner.initialize()
        
        # Run test suites
        await runner.test_basic_knowledge()
        await runner.test_programming_knowledge()
        await runner.test_reasoning_ability()
        await runner.test_creative_capabilities()
        await runner.test_model_comparison()
        await runner.test_consensus_mechanism()
        
        # Generate report
        runner.generate_report()
        
    except Exception as e:
        print(f"\n❌ Critical error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())