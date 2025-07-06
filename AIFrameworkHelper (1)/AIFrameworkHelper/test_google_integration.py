#!/usr/bin/env python3
"""
Test Google AI Integration
Quick test to verify Gemini models are properly integrated
"""

import asyncio
import os
from core.framework import AIOrchestrationFramework
from config import Config
from app import app

async def test_google_integration():
    """Test Google AI integration with Gemini models"""
    print("Testing Google AI Integration...")
    
    # Initialize framework within Flask app context
    with app.app_context():
        try:
            config = Config()
            framework = AIOrchestrationFramework(config)
            
            # Check if Google provider is available
            available_providers = framework.get_available_providers()
            print(f"Available providers: {available_providers}")
            
            if 'google' in available_providers:
                print("✓ Google AI provider is available")
                
                # Test basic request
                print("\nTesting basic request to Google AI...")
                try:
                    result = await framework.process_request(
                        prompt="Hello, please respond with a brief greeting.",
                        task_type="general",
                        user_id="test_user"
                    )
                    
                    print(f"✓ Request successful!")
                    print(f"Response: {result.response[:100]}...")
                    print(f"Metadata: {result.metadata}")
                    if result.consensus_result:
                        print(f"Consensus confidence: {result.consensus_result.confidence:.2f}")
                        print(f"Individual responses: {len(result.consensus_result.individual_responses)}")
                    
                except Exception as e:
                    print(f"✗ Request failed: {e}")
                    
            else:
                print("✗ Google AI provider not available")
                if not config.models['google'].api_key:
                    print("  Reason: No GEMINI_API_KEY found in environment")
                else:
                    print("  Reason: Unknown initialization error")
                    
        except Exception as e:
            print(f"✗ Framework initialization failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_google_integration())