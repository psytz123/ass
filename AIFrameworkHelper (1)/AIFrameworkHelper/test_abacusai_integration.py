#!/usr/bin/env python3
"""
Test AbacusAI Integration
Quick test to verify AbacusAI SDK is properly integrated
"""

import asyncio
import os
from core.framework import AIOrchestrationFramework
from config import Config
from app import app

async def test_abacusai_integration():
    """Test AbacusAI integration setup"""
    print("Testing AbacusAI Integration...")
    
    # Initialize framework within Flask app context
    with app.app_context():
        try:
            config = Config()
            framework = AIOrchestrationFramework(config)
            
            # Check if AbacusAI provider is available
            available_providers = framework.get_available_providers()
            print(f"Available providers: {available_providers}")
            
            if 'abacusai' in available_providers:
                print("✓ AbacusAI provider is available")
                
                # Test connection validation (without deployment)
                print("\nTesting AbacusAI connection validation...")
                abacus_connector = framework.connectors.get('abacusai')
                if abacus_connector:
                    try:
                        is_valid = await abacus_connector.validate_connection()
                        if is_valid:
                            print("✓ AbacusAI connection is valid")
                            
                            # List available deployments
                            print("\nListing available deployments...")
                            deployments = await abacus_connector.list_deployments()
                            print(f"Found {len(deployments)} deployments")
                            
                            if deployments:
                                for i, deployment in enumerate(deployments[:3]):  # Show first 3
                                    print(f"  {i+1}. {deployment.deployment_id} - {deployment.name}")
                            else:
                                print("  No deployments found. You'll need to deploy a model first.")
                        else:
                            print("✗ AbacusAI connection validation failed")
                    except Exception as e:
                        print(f"✗ AbacusAI connection test failed: {e}")
                        print("  This is expected if no API key is configured")
                else:
                    print("✗ AbacusAI connector not found")
                    
            else:
                print("✗ AbacusAI provider not available")
                if not config.models['abacusai'].api_key:
                    print("  Reason: No ABACUSAI_API_KEY found in environment")
                else:
                    print("  Reason: Unknown initialization error")
                    
            # Show supported models
            print(f"\nSupported AbacusAI models: {framework.connectors.get('abacusai', type('', (), {'get_supported_models': lambda: ['custom-model']})).get_supported_models()}")
                    
        except Exception as e:
            print(f"✗ Framework initialization failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_abacusai_integration())