#!/usr/bin/env python3
"""
Simple validation of AI Framework functionality
"""

import os
import sys
from app import app, db
from config import Config

def validate_framework():
    """Validate basic framework setup"""
    print("="*60)
    print("AI ORCHESTRATION FRAMEWORK VALIDATION")
    print("="*60)
    
    results = []
    
    # Test 1: Environment Check
    print("\n1. Checking Environment Variables...")
    api_keys = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
        "ABACUSAI_API_KEY": os.environ.get("ABACUSAI_API_KEY")
    }
    
    available_keys = []
    for key_name, key_value in api_keys.items():
        if key_value:
            print(f"  ✓ {key_name} is configured")
            available_keys.append(key_name)
        else:
            print(f"  ✗ {key_name} is not configured")
    
    results.append(("Environment Variables", len(available_keys) >= 2))
    
    # Test 2: Database Connection
    print("\n2. Checking Database Connection...")
    try:
        with app.app_context():
            # Try to query the database
            from models import Conversation
            count = db.session.query(Conversation).count()
            print(f"  ✓ Database connected (conversations: {count})")
            results.append(("Database Connection", True))
    except Exception as e:
        print(f"  ✗ Database error: {str(e)}")
        results.append(("Database Connection", False))
    
    # Test 3: Configuration
    print("\n3. Checking Configuration...")
    try:
        config = Config()
        print(f"  ✓ Configuration loaded")
        print(f"    - OpenAI model: {config.models['openai'].default_model}")
        print(f"    - Anthropic model: {config.models['anthropic'].default_model}")
        print(f"    - Routing strategy: {config.routing.default_strategy}")
        results.append(("Configuration", True))
    except Exception as e:
        print(f"  ✗ Configuration error: {str(e)}")
        results.append(("Configuration", False))
    
    # Test 4: Core Components
    print("\n4. Checking Core Components...")
    components = []
    try:
        from core.framework import AIOrchestrationFramework
        components.append("Framework")
        
        from core.model_connectors import OpenAIConnector, AnthropicConnector
        components.append("Model Connectors")
        
        from core.task_router import TaskRouter
        components.append("Task Router")
        
        from core.consensus import ConsensusEngine
        components.append("Consensus Engine")
        
        from core.memory import MemoryManager
        components.append("Memory Manager")
        
        print(f"  ✓ All core components available: {', '.join(components)}")
        results.append(("Core Components", True))
    except Exception as e:
        print(f"  ✗ Component error: {str(e)}")
        results.append(("Core Components", False))
    
    # Test 5: Model Connectors
    print("\n5. Checking Model Connectors...")
    try:
        with app.app_context():
            from core.framework import AIOrchestrationFramework
            framework = AIOrchestrationFramework(config)
            
            connectors = []
            # Check internal model_connectors attribute
            if hasattr(framework, '_model_connectors'):
                connectors = list(framework._model_connectors.keys())
            elif hasattr(framework, 'model_connectors'):
                connectors = list(framework.model_connectors.keys())
            
            # Also check individual connectors
            available_providers = []
            if hasattr(framework, 'openai_connector') and framework.openai_connector:
                available_providers.append('openai')
            if hasattr(framework, 'anthropic_connector') and framework.anthropic_connector:
                available_providers.append('anthropic')
            if hasattr(framework, 'google_connector') and framework.google_connector:
                available_providers.append('google')
            if hasattr(framework, 'abacusai_connector') and framework.abacusai_connector:
                available_providers.append('abacusai')
            
            if connectors or available_providers:
                all_connectors = set([c.value if hasattr(c, 'value') else str(c) for c in connectors])
                all_connectors.update(available_providers)
                print(f"  ✓ Active connectors: {list(all_connectors)}")
                results.append(("Model Connectors", True))
            else:
                print("  ✗ No active model connectors found")
                results.append(("Model Connectors", False))
    except Exception as e:
        print(f"  ✗ Connector error: {str(e)}")
        results.append(("Model Connectors", False))
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({(passed/total)*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Framework is ready for knowledge testing.")
    elif passed >= total * 0.8:
        print("\n✅ FRAMEWORK IS FUNCTIONAL. Most components are working.")
    else:
        print("\n⚠️ FRAMEWORK NEEDS ATTENTION. Some components are not working properly.")
    
    # Recommendations
    if len(available_keys) < 2:
        print("\n📝 Recommendation: Configure more API keys for better multi-model support.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    validate_framework()