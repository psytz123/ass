# ============================================================================
# AI ASSISTANT INTEGRATION & KNOWLEDGE VALIDATION TESTS
# ============================================================================

import pytest
import asyncio
import json
import re
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime
import difflib

# Import framework components
from core.framework import AIOrchestrationFramework
from core.types import ModelProvider, ModelRequest, ModelResponse
from core.consensus import ConsensusStrategy

# ============================================================================
# KNOWLEDGE VALIDATION FRAMEWORK
# ============================================================================

class KnowledgeDomain(str, Enum):
    """Different knowledge domains to test"""
    PROGRAMMING = "programming"
    MATHEMATICS = "mathematics"
    SCIENCE = "science"
    HISTORY = "history"
    LITERATURE = "literature"
    BUSINESS = "business"
    CURRENT_EVENTS = "current_events"
    REASONING = "reasoning"
    CREATIVITY = "creativity"
    ETHICS = "ethics"

@dataclass
class KnowledgeTestCase:
    """Structure for knowledge test cases"""
    question: str
    domain: KnowledgeDomain
    difficulty: str  # "basic", "intermediate", "advanced"
    expected_keywords: List[str]  # Keywords that should appear in correct answers
    incorrect_keywords: List[str]  # Keywords that suggest incorrect answers
    validation_func: Optional[callable] = None  # Custom validation function
    min_response_length: int = 50
    max_response_length: int = 2000

@dataclass
class ModelKnowledgeScore:
    """Score structure for model knowledge assessment"""
    model_name: str
    provider: ModelProvider
    domain_scores: Dict[KnowledgeDomain, float]
    overall_score: float
    response_quality: float
    consistency_score: float
    factual_accuracy: float
    reasoning_ability: float

class KnowledgeValidator:
    """Validates AI model responses for accuracy and quality"""
    
    def __init__(self):
        self.test_cases = self._initialize_test_cases()
    
    def _initialize_test_cases(self) -> Dict[KnowledgeDomain, List[KnowledgeTestCase]]:
        """Initialize comprehensive test cases across domains"""
        return {
            KnowledgeDomain.PROGRAMMING: [
                KnowledgeTestCase(
                    question="Explain the difference between recursion and iteration in programming",
                    domain=KnowledgeDomain.PROGRAMMING,
                    difficulty="intermediate",
                    expected_keywords=["function calls itself", "loop", "base case", "stack", "performance"],
                    incorrect_keywords=["identical", "no difference"]
                ),
                KnowledgeTestCase(
                    question="What is Big O notation and why is it important?",
                    domain=KnowledgeDomain.PROGRAMMING,
                    difficulty="intermediate",
                    expected_keywords=["time complexity", "algorithm", "efficiency", "worst case", "O(n)"],
                    incorrect_keywords=["big zero", "unimportant"]
                ),
                KnowledgeTestCase(
                    question="Write a Python function to find the factorial of a number using recursion",
                    domain=KnowledgeDomain.PROGRAMMING,
                    difficulty="basic",
                    expected_keywords=["def", "factorial", "return", "if", "else", "*"],
                    incorrect_keywords=["import math", "math.factorial"],
                    validation_func=lambda x: "def" in x and "return" in x and "*" in x
                ),
                KnowledgeTestCase(
                    question="Explain the SOLID principles in object-oriented programming",
                    domain=KnowledgeDomain.PROGRAMMING,
                    difficulty="advanced",
                    expected_keywords=["Single Responsibility", "Open/Closed", "Liskov", "Interface", "Dependency"],
                    incorrect_keywords=["LIQUID", "only one principle"]
                )
            ],
            
            KnowledgeDomain.MATHEMATICS: [
                KnowledgeTestCase(
                    question="What is calculus and what are its main branches?",
                    domain=KnowledgeDomain.MATHEMATICS,
                    difficulty="intermediate",
                    expected_keywords=["derivatives", "integrals", "differential", "integral calculus", "limits"],
                    incorrect_keywords=["algebra", "geometry only"]
                ),
                KnowledgeTestCase(
                    question="Solve: What is the derivative of x^2 + 3x + 5?",
                    domain=KnowledgeDomain.MATHEMATICS,
                    difficulty="basic",
                    expected_keywords=["2x + 3", "2x", "+3"],
                    incorrect_keywords=["x^2", "3x + 5"],
                    validation_func=lambda x: "2x" in x.replace(" ", "") and "+3" in x.replace(" ", "")
                ),
                KnowledgeTestCase(
                    question="Explain the Pythagorean theorem and provide its formula",
                    domain=KnowledgeDomain.MATHEMATICS,
                    difficulty="basic",
                    expected_keywords=["a² + b² = c²", "right triangle", "hypotenuse", "squared"],
                    incorrect_keywords=["a + b = c", "area formula"]
                ),
                KnowledgeTestCase(
                    question="What is the fundamental theorem of calculus?",
                    domain=KnowledgeDomain.MATHEMATICS,
                    difficulty="advanced",
                    expected_keywords=["derivative", "integral", "antiderivative", "continuous function"],
                    incorrect_keywords=["algebra", "geometry"]
                )
            ],
            
            KnowledgeDomain.SCIENCE: [
                KnowledgeTestCase(
                    question="Explain photosynthesis and its importance",
                    domain=KnowledgeDomain.SCIENCE,
                    difficulty="intermediate",
                    expected_keywords=["chlorophyll", "sunlight", "carbon dioxide", "oxygen", "glucose", "plants"],
                    incorrect_keywords=["respiration only", "animals"]
                ),
                KnowledgeTestCase(
                    question="What are the three laws of thermodynamics?",
                    domain=KnowledgeDomain.SCIENCE,
                    difficulty="advanced",
                    expected_keywords=["energy conservation", "entropy", "absolute zero", "first law", "second law"],
                    incorrect_keywords=["four laws", "newton's laws"]
                ),
                KnowledgeTestCase(
                    question="Describe the structure of an atom",
                    domain=KnowledgeDomain.SCIENCE,
                    difficulty="basic",
                    expected_keywords=["nucleus", "protons", "neutrons", "electrons", "charge"],
                    incorrect_keywords=["molecules", "compounds only"]
                )
            ],
            
            KnowledgeDomain.HISTORY: [
                KnowledgeTestCase(
                    question="What were the main causes of World War I?",
                    domain=KnowledgeDomain.HISTORY,
                    difficulty="intermediate",
                    expected_keywords=["assassination", "Franz Ferdinand", "alliances", "imperialism", "nationalism"],
                    incorrect_keywords=["Hitler", "World War II", "Pearl Harbor"]
                ),
                KnowledgeTestCase(
                    question="Who was the first President of the United States?",
                    domain=KnowledgeDomain.HISTORY,
                    difficulty="basic",
                    expected_keywords=["George Washington"],
                    incorrect_keywords=["Thomas Jefferson", "John Adams", "Benjamin Franklin"]
                ),
                KnowledgeTestCase(
                    question="Explain the significance of the Magna Carta",
                    domain=KnowledgeDomain.HISTORY,
                    difficulty="intermediate",
                    expected_keywords=["1215", "King John", "limited monarchy", "rule of law", "barons"],
                    incorrect_keywords=["democracy", "parliament", "American"]
                )
            ],
            
            KnowledgeDomain.BUSINESS: [
                KnowledgeTestCase(
                    question="What is the difference between revenue and profit?",
                    domain=KnowledgeDomain.BUSINESS,
                    difficulty="basic",
                    expected_keywords=["total income", "expenses", "costs", "net income", "gross"],
                    incorrect_keywords=["same thing", "identical"]
                ),
                KnowledgeTestCase(
                    question="Explain the concept of supply and demand",
                    domain=KnowledgeDomain.BUSINESS,
                    difficulty="intermediate",
                    expected_keywords=["price", "quantity", "equilibrium", "market", "economics"],
                    incorrect_keywords=["unrelated", "constant price"]
                ),
                KnowledgeTestCase(
                    question="What is a SWOT analysis and when is it used?",
                    domain=KnowledgeDomain.BUSINESS,
                    difficulty="intermediate",
                    expected_keywords=["Strengths", "Weaknesses", "Opportunities", "Threats", "strategic planning"],
                    incorrect_keywords=["SPOT", "financial only"]
                )
            ],
            
            KnowledgeDomain.REASONING: [
                KnowledgeTestCase(
                    question="If all birds can fly, and penguins are birds, can penguins fly? Explain your reasoning.",
                    domain=KnowledgeDomain.REASONING,
                    difficulty="intermediate",
                    expected_keywords=["premise false", "not all birds", "penguins cannot fly", "exception"],
                    incorrect_keywords=["yes, penguins fly", "all birds fly"]
                ),
                KnowledgeTestCase(
                    question="A farmer has 15 cows. All but 8 die. How many are left?",
                    domain=KnowledgeDomain.REASONING,
                    difficulty="basic",
                    expected_keywords=["8", "eight"],
                    incorrect_keywords=["7", "15", "23"],
                    validation_func=lambda x: any(word in x.lower() for word in ["8", "eight"])
                ),
                KnowledgeTestCase(
                    question="Explain the trolley problem and its ethical implications",
                    domain=KnowledgeDomain.REASONING,
                    difficulty="advanced",
                    expected_keywords=["utilitarian", "deontological", "moral dilemma", "ethics", "consequences"],
                    incorrect_keywords=["simple answer", "obvious choice"]
                )
            ],
            
            KnowledgeDomain.CREATIVITY: [
                KnowledgeTestCase(
                    question="Write a creative short story opening about a time traveler",
                    domain=KnowledgeDomain.CREATIVITY,
                    difficulty="intermediate",
                    expected_keywords=["time", "travel", "narrative", "character"],
                    incorrect_keywords=["technical manual", "factual report"],
                    min_response_length=100,
                    validation_func=lambda x: len(x) > 100 and any(word in x.lower() for word in ["time", "travel"])
                ),
                KnowledgeTestCase(
                    question="Suggest 5 innovative uses for a paperclip",
                    domain=KnowledgeDomain.CREATIVITY,
                    difficulty="basic",
                    expected_keywords=["creative", "innovative", "alternative"],
                    incorrect_keywords=["only paper", "traditional use"],
                    validation_func=lambda x: len([line for line in x.split('\n') if line.strip()]) >= 3
                )
            ]
        }
    
    def validate_response(self, response: str, test_case: KnowledgeTestCase) -> Dict[str, Any]:
        """Validate a response against a test case"""
        response_lower = response.lower()
        
        # Check expected keywords
        expected_found = sum(1 for keyword in test_case.expected_keywords 
                           if keyword.lower() in response_lower)
        expected_score = expected_found / len(test_case.expected_keywords) if test_case.expected_keywords else 1.0
        
        # Check for incorrect keywords (negative scoring)
        incorrect_found = sum(1 for keyword in test_case.incorrect_keywords 
                            if keyword.lower() in response_lower)
        incorrect_penalty = incorrect_found * 0.2  # 20% penalty per incorrect keyword
        
        # Length validation
        length_appropriate = test_case.min_response_length <= len(response) <= test_case.max_response_length
        
        # Custom validation if provided
        custom_valid = True
        if test_case.validation_func:
            try:
                custom_valid = test_case.validation_func(response)
            except:
                custom_valid = False
        
        # Calculate overall score
        base_score = expected_score - incorrect_penalty
        length_bonus = 0.1 if length_appropriate else -0.1
        custom_bonus = 0.1 if custom_valid else -0.2
        
        final_score = max(0.0, min(1.0, base_score + length_bonus + custom_bonus))
        
        return {
            "score": final_score,
            "expected_keywords_found": expected_found,
            "total_expected_keywords": len(test_case.expected_keywords),
            "incorrect_keywords_found": incorrect_found,
            "length_appropriate": length_appropriate,
            "custom_validation_passed": custom_valid,
            "response_length": len(response),
            "detailed_breakdown": {
                "keyword_score": expected_score,
                "incorrect_penalty": incorrect_penalty,
                "length_bonus": length_bonus,
                "custom_bonus": custom_bonus
            }
        }

# ============================================================================
# AI ASSISTANT INTEGRATION TESTS
# ============================================================================

class TestAIAssistantIntegration:
    """Test AI assistant integration and knowledge capabilities"""
    
    @pytest.fixture
    async def knowledge_framework(self, test_framework):
        """Framework setup for knowledge testing"""
        # Ensure all providers are available for comparison
        return test_framework
    
    @pytest.fixture
    def knowledge_validator(self):
        """Knowledge validation helper"""
        return KnowledgeValidator()
    
    async def test_basic_model_integration(self, knowledge_framework):
        """Test that all AI models integrate correctly with the framework"""
        test_prompt = "What is artificial intelligence?"
        
        # Test each provider individually
        for provider in [ModelProvider.OPENAI, ModelProvider.ANTHROPIC, ModelProvider.GOOGLE]:
            try:
                result = await knowledge_framework.process_request(
                    prompt=test_prompt,
                    task_type="knowledge_query",
                    user_id="integration_test",
                    provider_override=provider  # Force specific provider
                )
                
                assert result["response"] is not None
                assert len(result["response"]) > 50  # Substantial response
                assert provider in result["metadata"]["providers_used"]
                
            except Exception as e:
                pytest.fail(f"Provider {provider} failed integration test: {e}")
    
    async def test_response_consistency_across_models(self, knowledge_framework):
        """Test consistency of responses across different AI models"""
        factual_questions = [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What is 2 + 2?",
            "What is the chemical symbol for water?"
        ]
        
        for question in factual_questions:
            responses = {}
            
            # Get responses from all providers
            for provider in [ModelProvider.OPENAI, ModelProvider.ANTHROPIC, ModelProvider.GOOGLE]:
                try:
                    result = await knowledge_framework.process_request(
                        prompt=question,
                        task_type="factual_query",
                        user_id="consistency_test",
                        provider_override=provider
                    )
                    responses[provider] = result["response"].lower()
                except:
                    continue
            
            # Check for basic consistency in factual answers
            if len(responses) >= 2:
                # For capital of France
                if "france" in question.lower():
                    assert all("paris" in response for response in responses.values()), \
                        f"Inconsistent responses for '{question}': {responses}"
                
                # For Romeo and Juliet
                if "romeo" in question.lower():
                    assert all("shakespeare" in response for response in responses.values()), \
                        f"Inconsistent responses for '{question}': {responses}"
                
                # For 2 + 2
                if "2 + 2" in question:
                    assert all(any(num in response for num in ["4", "four"]) for response in responses.values()), \
                        f"Inconsistent responses for '{question}': {responses}"
    
    async def test_domain_specific_knowledge(self, knowledge_framework, knowledge_validator):
        """Test knowledge across different domains"""
        domain_scores = {}
        
        for domain, test_cases in knowledge_validator.test_cases.items():
            domain_score = 0.0
            valid_tests = 0
            
            for test_case in test_cases[:2]:  # Test first 2 cases per domain
                try:
                    result = await knowledge_framework.process_request(
                        prompt=test_case.question,
                        task_type="knowledge_query",
                        user_id="domain_test"
                    )
                    
                    validation_result = knowledge_validator.validate_response(
                        result["response"], test_case
                    )
                    
                    domain_score += validation_result["score"]
                    valid_tests += 1
                    
                except Exception as e:
                    print(f"Failed test case in {domain}: {e}")
                    continue
            
            if valid_tests > 0:
                domain_scores[domain] = domain_score / valid_tests
        
        # Ensure reasonable performance across domains
        for domain, score in domain_scores.items():
            assert score >= 0.3, f"Poor performance in {domain}: {score}"
        
        print(f"Domain knowledge scores: {domain_scores}")
    
    async def test_programming_knowledge_depth(self, knowledge_framework, knowledge_validator):
        """Deep test of programming knowledge and code generation"""
        programming_tests = [
            {
                "prompt": "Write a Python function to implement binary search",
                "validation": lambda x: all(keyword in x.lower() for keyword in ["def", "binary", "search", "return"]),
                "complexity": "intermediate"
            },
            {
                "prompt": "Explain the difference between '==' and 'is' in Python",
                "validation": lambda x: "identity" in x.lower() and "value" in x.lower(),
                "complexity": "intermediate"
            },
            {
                "prompt": "Write a SQL query to find the second highest salary",
                "validation": lambda x: "select" in x.lower() and ("limit" in x.lower() or "top" in x.lower()),
                "complexity": "intermediate"
            },
            {
                "prompt": "Implement a simple REST API endpoint in Python using Flask",
                "validation": lambda x: "flask" in x.lower() and "route" in x.lower() and "def" in x.lower(),
                "complexity": "advanced"
            }
        ]
        
        programming_scores = []
        
        for test in programming_tests:
            result = await knowledge_framework.process_request(
                prompt=test["prompt"],
                task_type="code_generation",
                user_id="programming_test"
            )
            
            response = result["response"]
            
            # Basic validation
            assert len(response) > 100, f"Response too short for: {test['prompt']}"
            
            # Custom validation
            if test["validation"](response):
                programming_scores.append(1.0)
            else:
                programming_scores.append(0.5)  # Partial credit
            
            # Code-specific checks
            if "python" in test["prompt"].lower():
                assert "def " in response or "class " in response, \
                    f"Python code should contain function/class definition"
            
            if "sql" in test["prompt"].lower():
                assert "select" in response.lower(), \
                    f"SQL query should contain SELECT statement"
        
        avg_programming_score = sum(programming_scores) / len(programming_scores)
        assert avg_programming_score >= 0.7, f"Programming knowledge score too low: {avg_programming_score}"
    
    async def test_reasoning_and_logic(self, knowledge_framework):
        """Test logical reasoning capabilities"""
        logic_tests = [
            {
                "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                "expected_answer": "5",
                "reasoning_required": True
            },
            {
                "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                "expected_answer": "0.05",
                "reasoning_required": True
            },
            {
                "prompt": "You have a 3-gallon jug and a 5-gallon jug. How can you measure exactly 4 gallons?",
                "expected_keywords": ["fill", "pour", "empty", "measure"],
                "reasoning_required": True
            }
        ]
        
        reasoning_scores = []
        
        for test in logic_tests:
            result = await knowledge_framework.process_request(
                prompt=test["prompt"],
                task_type="reasoning_task",
                user_id="reasoning_test"
            )
            
            response = result["response"].lower()
            
            if "expected_answer" in test:
                if test["expected_answer"] in response:
                    reasoning_scores.append(1.0)
                else:
                    reasoning_scores.append(0.0)
            
            if "expected_keywords" in test:
                found_keywords = sum(1 for keyword in test["expected_keywords"] 
                                   if keyword in response)
                keyword_score = found_keywords / len(test["expected_keywords"])
                reasoning_scores.append(keyword_score)
            
            # Check for explanation/reasoning
            reasoning_indicators = ["because", "therefore", "since", "step", "first", "then"]
            has_reasoning = any(indicator in response for indicator in reasoning_indicators)
            
            if test["reasoning_required"]:
                assert has_reasoning, f"Response lacks reasoning for: {test['prompt']}"
        
        avg_reasoning_score = sum(reasoning_scores) / len(reasoning_scores)
        assert avg_reasoning_score >= 0.6, f"Reasoning score too low: {avg_reasoning_score}"
    
    async def test_creative_capabilities(self, knowledge_framework):
        """Test creative and generative capabilities"""
        creative_tests = [
            {
                "prompt": "Write a haiku about artificial intelligence",
                "validation": lambda x: len(x.split('\n')) >= 3,
                "type": "poetry"
            },
            {
                "prompt": "Create a short dialogue between a robot and a human about friendship",
                "validation": lambda x: "robot" in x.lower() and "human" in x.lower(),
                "type": "dialogue"
            },
            {
                "prompt": "Brainstorm 5 creative solutions to reduce plastic waste",
                "validation": lambda x: len([line for line in x.split('\n') if line.strip()]) >= 3,
                "type": "brainstorming"
            }
        ]
        
        for test in creative_tests:
            result = await knowledge_framework.process_request(
                prompt=test["prompt"],
                task_type="creative_task",
                user_id="creative_test"
            )
            
            response = result["response"]
            
            # Basic length check
            assert len(response) > 50, f"Creative response too short: {test['prompt']}"
            
            # Custom validation
            assert test["validation"](response), f"Creative validation failed for: {test['prompt']}"
            
            # Creativity indicators
            creative_indicators = ["creative", "innovative", "unique", "original", "imagine"]
            # Note: Response doesn't need to contain these words, but the content should be creative
    
    async def test_knowledge_boundary_detection(self, knowledge_framework):
        """Test how well the AI recognizes its knowledge limitations"""
        boundary_tests = [
            {
                "prompt": "What will the stock price of Apple be next week?",
                "should_express_uncertainty": True,
                "uncertainty_keywords": ["cannot", "predict", "uncertain", "unknown", "don't know"]
            },
            {
                "prompt": "What is my personal favorite color?",
                "should_express_uncertainty": True,
                "uncertainty_keywords": ["don't know", "cannot know", "no way", "impossible"]
            },
            {
                "prompt": "What happened in the news yesterday?",
                "should_express_uncertainty": True,
                "uncertainty_keywords": ["don't have", "cannot access", "no real-time", "cutoff"]
            }
        ]
        
        for test in boundary_tests:
            result = await knowledge_framework.process_request(
                prompt=test["prompt"],
                task_type="boundary_test",
                user_id="boundary_test"
            )
            
            response = result["response"].lower()
            
            if test["should_express_uncertainty"]:
                uncertainty_expressed = any(keyword in response for keyword in test["uncertainty_keywords"])
                assert uncertainty_expressed, f"Failed to express uncertainty for: {test['prompt']}\nResponse: {response}"
    
    async def test_multi_step_reasoning(self, knowledge_framework):
        """Test complex multi-step reasoning tasks"""
        complex_tasks = [
            {
                "prompt": """
                A company has 100 employees. 60% work in engineering, 25% in sales, and 15% in marketing.
                If the company grows by 50% and maintains the same ratios, how many new engineering employees will be hired?
                Show your work step by step.
                """,
                "expected_steps": ["calculate current", "calculate new total", "calculate new engineering", "subtract"],
                "expected_answer": "30"
            },
            {
                "prompt": """
                Plan a route for a delivery truck that needs to visit 5 stores in optimal order.
                The stores are: A (downtown), B (north), C (east), D (south), E (west).
                The truck starts at the warehouse (center). What factors would you consider?
                """,
                "expected_factors": ["distance", "traffic", "time", "efficiency", "route optimization"]
            }
        ]
        
        for task in complex_tasks:
            result = await knowledge_framework.process_request(
                prompt=task["prompt"],
                task_type="complex_reasoning",
                user_id="complex_test"
            )
            
            response = result["response"].lower()
            
            # Check for step-by-step reasoning
            step_indicators = ["step", "first", "next", "then", "finally", "1.", "2.", "3."]
            has_steps = any(indicator in response for indicator in step_indicators)
            assert has_steps, f"Multi-step task lacks step-by-step reasoning: {task['prompt']}"
            
            # Check for expected content
            if "expected_steps" in task:
                found_steps = sum(1 for step in task["expected_steps"] if step in response)
                step_ratio = found_steps / len(task["expected_steps"])
                assert step_ratio >= 0.5, f"Missing key reasoning steps: {step_ratio}"
            
            if "expected_answer" in task:
                assert task["expected_answer"] in response, f"Missing expected answer: {task['expected_answer']}"
            
            if "expected_factors" in task:
                found_factors = sum(1 for factor in task["expected_factors"] if factor in response)
                factor_ratio = found_factors / len(task["expected_factors"])
                assert factor_ratio >= 0.4, f"Missing key factors: {factor_ratio}"

# ============================================================================
# KNOWLEDGE COMPARISON TESTS
# ============================================================================

class TestKnowledgeComparison:
    """Compare knowledge quality across different AI models"""
    
    async def test_cross_model_knowledge_quality(self, knowledge_framework, knowledge_validator):
        """Compare knowledge quality across different models"""
        comparison_questions = [
            "Explain quantum computing and its potential applications",
            "What are the key principles of machine learning?",
            "Describe the process of photosynthesis in detail",
            "Explain the causes and effects of climate change"
        ]
        
        model_scores = {provider: [] for provider in ModelProvider}
        
        for question in comparison_questions:
            provider_responses = {}
            
            # Get response from each provider
            for provider in [ModelProvider.OPENAI, ModelProvider.ANTHROPIC, ModelProvider.GOOGLE]:
                try:
                    result = await knowledge_framework.process_request(
                        prompt=question,
                        task_type="knowledge_comparison",
                        user_id="comparison_test",
                        provider_override=provider
                    )
                    provider_responses[provider] = result["response"]
                except:
                    continue
            
            # Score responses based on multiple criteria
            for provider, response in provider_responses.items():
                score = self._score_response_quality(response, question)
                model_scores[provider].append(score)
        
        # Calculate average scores
        avg_scores = {}
        for provider, scores in model_scores.items():
            if scores:
                avg_scores[provider] = sum(scores) / len(scores)
        
        print(f"Cross-model knowledge quality scores: {avg_scores}")
        
        # Ensure all models perform reasonably
        for provider, score in avg_scores.items():
            assert score >= 0.4, f"Model {provider} has poor knowledge quality: {score}"
    
    def _score_response_quality(self, response: str, question: str) -> float:
        """Score response quality based on multiple factors"""
        score = 0.0
        
        # Length appropriateness (not too short, not too verbose)
        length = len(response)
        if 100 <= length <= 1000:
            score += 0.2
        elif 50 <= length <= 1500:
            score += 0.1
        
        # Structure indicators
        structure_indicators = ["\n", ":", ".", "first", "second", "however", "therefore"]
        structure_score = sum(0.05 for indicator in structure_indicators if indicator in response.lower())
        score += min(0.3, structure_score)
        
        # Domain-specific quality
        if "quantum" in question.lower():
            quality_indicators = ["qubit", "superposition", "entanglement", "computation"]
            quality_score = sum(0.1 for indicator in quality_indicators if indicator in response.lower())
            score += min(0.3, quality_score)
        
        elif "machine learning" in question.lower():
            quality_indicators = ["algorithm", "data", "training", "model", "prediction"]
            quality_score = sum(0.1 for indicator in quality_indicators if indicator in response.lower())
            score += min(0.3, quality_score)
        
        elif "photosynthesis" in question.lower():
            quality_indicators = ["chlorophyll", "sunlight", "carbon dioxide", "oxygen", "glucose"]
            quality_score = sum(0.1 for indicator in quality_indicators if indicator in response.lower())
            score += min(0.3, quality_score)
        
        # Factual accuracy indicators (basic check)
        factual_problems = ["incorrect", "wrong", "false", "myth"]
        if not any(problem in response.lower() for problem in factual_problems):
            score += 0.2
        
        return min(1.0, score)
    
    async def test_consensus_quality_with_knowledge_validation(self, knowledge_framework, knowledge_validator):
        """Test consensus quality specifically for knowledge-based tasks"""
        knowledge_questions = [
            "What are the fundamental forces in physics?",
            "Explain the structure and function of DNA",
            "What are the main economic theories?",
            "Describe the water cycle"
        ]
        
        for question in knowledge_questions:
            # Get consensus response
            result = await knowledge_framework.process_request(
                prompt=question,
                task_type="knowledge_query",
                user_id="consensus_knowledge_test",
                require_consensus=True,
                min_providers=2
            )
            
            response = result["response"]
            metadata = result["metadata"]
            
            # Ensure consensus was actually applied
            assert metadata.get("consensus_applied", False), "Consensus should be applied for knowledge tasks"
            assert len(metadata["providers_used"]) >= 2, "Should use multiple providers for consensus"
            
            # Validate consensus response quality
            consensus_score = self._score_response_quality(response, question)
            assert consensus_score >= 0.5, f"Consensus response quality too low: {consensus_score}"
            
            # Check for consensus confidence
            if "confidence" in metadata:
                assert metadata["confidence"] >= 0.3, "Consensus confidence too low"

# ============================================================================
# SPECIALIZED KNOWLEDGE TESTS
# ============================================================================

class TestSpecializedKnowledge:
    """Test specialized domain knowledge and expert-level capabilities"""
    
    async def test_technical_accuracy(self, knowledge_framework):
        """Test technical accuracy in specialized domains"""
        technical_tests = [
            {
                "domain": "software_engineering",
                "question": "Explain the difference between microservices and monolithic architecture",
                "required_concepts": ["scalability", "deployment", "services", "communication"],
                "accuracy_keywords": ["distributed", "independent", "API", "database"]
            },
            {
                "domain": "data_science",
                "question": "What is overfitting in machine learning and how can it be prevented?",
                "required_concepts": ["training", "validation", "generalization", "bias"],
                "accuracy_keywords": ["regularization", "cross-validation", "dropout", "early stopping"]
            },
            {
                "domain": "cybersecurity",
                "question": "Explain SQL injection attacks and prevention methods",
                "required_concepts": ["database", "input", "validation", "security"],
                "accuracy_keywords": ["prepared statements", "sanitization", "parameterized queries"]
            }
        ]
        
        for test in technical_tests:
            result = await knowledge_framework.process_request(
                prompt=test["question"],
                task_type="technical_query",
                user_id="technical_test"
            )
            
            response = result["response"].lower()
            
            # Check for required concepts
            concepts_found = sum(1 for concept in test["required_concepts"] 
                               if concept in response)
            concept_ratio = concepts_found / len(test["required_concepts"])
            assert concept_ratio >= 0.7, f"Missing key concepts in {test['domain']}: {concept_ratio}"
            
            # Check for technical accuracy
            accuracy_found = sum(1 for keyword in test["accuracy_keywords"] 
                               if keyword in response)
            accuracy_ratio = accuracy_found / len(test["accuracy_keywords"])
            assert accuracy_ratio >= 0.4, f"Low technical accuracy in {test['domain']}: {accuracy_ratio}"
    
    async def test_interdisciplinary_knowledge(self, knowledge_framework):
        """Test knowledge that spans multiple disciplines"""
        interdisciplinary_tests = [
            {
                "question": "How does computer vision relate to neuroscience and psychology?",
                "disciplines": ["computer science", "neuroscience", "psychology"],
                "connection_keywords": ["neural networks", "visual cortex", "perception", "cognition"]
            },
            {
                "question": "Explain the intersection of economics and environmental science in climate policy",
                "disciplines": ["economics", "environmental science", "policy"],
                "connection_keywords": ["externalities", "carbon pricing", "cost-benefit", "sustainability"]
            },
            {
                "question": "How do physics principles apply to medical imaging technologies?",
                "disciplines": ["physics", "medicine", "technology"],
                "connection_keywords": ["electromagnetic", "radiation", "waves", "imaging", "diagnosis"]
            }
        ]
        
        for test in interdisciplinary_tests:
            result = await knowledge_framework.process_request(
                prompt=test["question"],
                task_type="interdisciplinary_query",
                user_id="interdisciplinary_test"
            )
            
            response = result["response"].lower()
            
            # Check for interdisciplinary connections
            connections_found = sum(1 for keyword in test["connection_keywords"] 
                                  if keyword in response)
            connection_ratio = connections_found / len(test["connection_keywords"])
            assert connection_ratio >= 0.5, f"Weak interdisciplinary connections: {connection_ratio}"
            
            # Check for mention of multiple disciplines
            disciplines_mentioned = sum(1 for discipline in test["disciplines"] 
                                      if discipline in response)
            assert disciplines_mentioned >= 2, f"Should mention multiple disciplines: {disciplines_mentioned}"
    
    async def test_current_knowledge_boundaries(self, knowledge_framework):
        """Test awareness of knowledge cutoff and current limitations"""
        boundary_tests = [
            {
                "question": "What are the latest developments in AI research from 2024?",
                "should_acknowledge_cutoff": True
            },
            {
                "question": "Who won the 2024 Nobel Prize in Physics?",
                "should_acknowledge_cutoff": True
            },
            {
                "question": "What is the current price of Bitcoin?",
                "should_acknowledge_realtime": True
            },
            {
                "question": "What happened in the news today?",
                "should_acknowledge_realtime": True
            }
        ]
        
        for test in boundary_tests:
            result = await knowledge_framework.process_request(
                prompt=test["question"],
                task_type="boundary_test",
                user_id="boundary_test"
            )
            
            response = result["response"].lower()
            
            if test.get("should_acknowledge_cutoff"):
                cutoff_indicators = ["knowledge cutoff", "training data", "last update", "as of", "up to"]
                acknowledges_cutoff = any(indicator in response for indicator in cutoff_indicators)
                assert acknowledges_cutoff, f"Should acknowledge knowledge cutoff: {test['question']}"
            
            if test.get("should_acknowledge_realtime"):
                realtime_indicators = ["real-time", "current", "live", "up-to-date", "cannot access"]
                acknowledges_realtime = any(indicator in response for indicator in realtime_indicators)
                assert acknowledges_realtime, f"Should acknowledge real-time limitations: {test['question']}"

# ============================================================================
# PERFORMANCE BENCHMARKING FOR KNOWLEDGE TASKS
# ============================================================================

class TestKnowledgePerformance:
    """Test performance characteristics for knowledge-intensive tasks"""
    
    async def test_knowledge_query_latency(self, knowledge_framework):
        """Test response times for knowledge queries of varying complexity"""
        complexity_tests = [
            {
                "prompt": "What is Python?",
                "complexity": "simple",
                "max_latency_ms": 3000
            },
            {
                "prompt": "Explain object-oriented programming concepts with examples",
                "complexity": "medium",
                "max_latency_ms": 5000
            },
            {
                "prompt": "Compare and contrast different machine learning algorithms, their use cases, advantages, and limitations",
                "complexity": "complex",
                "max_latency_ms": 8000
            }
        ]
        
        for test in complexity_tests:
            start_time = time.time()
            
            result = await knowledge_framework.process_request(
                prompt=test["prompt"],
                task_type="knowledge_query",
                user_id="latency_test"
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            assert result["response"] is not None
            assert elapsed_ms <= test["max_latency_ms"], \
                f"Latency too high for {test['complexity']} query: {elapsed_ms}ms > {test['max_latency_ms']}ms"
    
    async def test_concurrent_knowledge_queries(self, knowledge_framework):
        """Test handling multiple concurrent knowledge queries"""
        concurrent_queries = [
            "What is machine learning?",
            "Explain quantum physics",
            "How does photosynthesis work?",
            "What is blockchain technology?",
            "Describe the solar system"
        ]
        
        async def make_query(prompt: str, query_id: int):
            return await knowledge_framework.process_request(
                prompt=prompt,
                task_type="knowledge_query",
                user_id=f"concurrent_user_{query_id}"
            )
        
        start_time = time.time()
        
        # Execute all queries concurrently
        tasks = [make_query(query, i) for i, query in enumerate(concurrent_queries)]
        results = await asyncio.gather(*tasks)
        
        elapsed_time = time.time() - start_time
        
        # All queries should succeed
        assert len(results) == len(concurrent_queries)
        assert all(result["response"] is not None for result in results)
        
        # Should be faster than sequential execution
        assert elapsed_time < 15.0, f"Concurrent execution too slow: {elapsed_time}s"
        
        # Responses should be unique (not cached artifacts)
        response_contents = [result["response"] for result in results]
        unique_responses = len(set(response_contents))
        assert unique_responses >= 3, "Responses should be varied for different questions"

# ============================================================================
# KNOWLEDGE INTEGRATION TEST SUITE
# ============================================================================

class TestKnowledgeIntegrationSuite:
    """Comprehensive knowledge integration test suite"""
    
    async def test_complete_knowledge_assessment(self, knowledge_framework, knowledge_validator):
        """Run complete knowledge assessment across all domains"""
        assessment_results = {}
        
        for domain, test_cases in knowledge_validator.test_cases.items():
            domain_results = []
            
            for test_case in test_cases:
                try:
                    # Process the knowledge query
                    result = await knowledge_framework.process_request(
                        prompt=test_case.question,
                        task_type="knowledge_assessment",
                        user_id="assessment_test"
                    )
                    
                    # Validate the response
                    validation = knowledge_validator.validate_response(
                        result["response"], test_case
                    )
                    
                    domain_results.append({
                        "question": test_case.question,
                        "difficulty": test_case.difficulty,
                        "score": validation["score"],
                        "validation_details": validation,
                        "response_length": len(result["response"]),
                        "providers_used": result["metadata"]["providers_used"]
                    })
                    
                except Exception as e:
                    domain_results.append({
                        "question": test_case.question,
                        "difficulty": test_case.difficulty,
                        "score": 0.0,
                        "error": str(e)
                    })
            
            assessment_results[domain] = domain_results
        
        # Generate comprehensive report
        self._generate_knowledge_report(assessment_results)
        
        # Ensure minimum performance across all domains
        for domain, results in assessment_results.items():
            valid_scores = [r["score"] for r in results if "error" not in r]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                assert avg_score >= 0.4, f"Poor performance in {domain}: {avg_score}"
    
    def _generate_knowledge_report(self, assessment_results: Dict[str, List[Dict]]):
        """Generate a comprehensive knowledge assessment report"""
        print("\n" + "="*80)
        print("KNOWLEDGE ASSESSMENT REPORT")
        print("="*80)
        
        overall_scores = []
        
        for domain, results in assessment_results.items():
            print(f"\n{domain.upper()} DOMAIN:")
            print("-" * 40)
            
            valid_results = [r for r in results if "error" not in r]
            if not valid_results:
                print("No valid results")
                continue
            
            scores = [r["score"] for r in valid_results]
            avg_score = sum(scores) / len(scores)
            overall_scores.extend(scores)
            
            print(f"Average Score: {avg_score:.2f}")
            print(f"Tests Completed: {len(valid_results)}/{len(results)}")
            
            # Difficulty breakdown
            by_difficulty = {}
            for result in valid_results:
                diff = result["difficulty"]
                if diff not in by_difficulty:
                    by_difficulty[diff] = []
                by_difficulty[diff].append(result["score"])
            
            for difficulty, scores in by_difficulty.items():
                avg_diff_score = sum(scores) / len(scores)
                print(f"  {difficulty.capitalize()}: {avg_diff_score:.2f} ({len(scores)} tests)")
        
        if overall_scores:
            overall_avg = sum(overall_scores) / len(overall_scores)
            print(f"\nOVERALL KNOWLEDGE SCORE: {overall_avg:.2f}")
            print(f"Total Tests: {len(overall_scores)}")
        
        print("="*80)

# ============================================================================
# EXAMPLE TEST EXECUTION COMMANDS
# ============================================================================

"""
# Run all knowledge and integration tests
pytest tests/test_ai_knowledge_integration.py -v

# Run only basic integration tests
pytest tests/test_ai_knowledge_integration.py::TestAIAssistantIntegration -v

# Run knowledge comparison tests
pytest tests/test_ai_knowledge_integration.py::TestKnowledgeComparison -v

# Run specialized knowledge tests
pytest tests/test_ai_knowledge_integration.py::TestSpecializedKnowledge -v

# Run performance tests for knowledge tasks
pytest tests/test_ai_knowledge_integration.py::TestKnowledgePerformance -v

# Run complete knowledge assessment
pytest tests/test_ai_knowledge_integration.py::TestKnowledgeIntegrationSuite::test_complete_knowledge_assessment -v -s

# Run with live API integration (requires API keys)
pytest tests/test_ai_knowledge_integration.py -m live_api --live-api

# Generate detailed knowledge report
pytest tests/test_ai_knowledge_integration.py::TestKnowledgeIntegrationSuite::test_complete_knowledge_assessment -v -s --tb=short

# Test specific knowledge domain
pytest tests/test_ai_knowledge_integration.py -k "programming" -v

# Test reasoning capabilities only
pytest tests/test_ai_knowledge_integration.py -k "reasoning" -v

# Performance benchmarking for knowledge tasks
pytest tests/test_ai_knowledge_integration.py::TestKnowledgePerformance -v --benchmark
"""