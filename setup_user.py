#!/usr/bin/env python3
"""
PromptGen MCP Setup Script
Helps users configure their local MCP server with the correct API keys
"""

import os
import json
import sys
from pathlib import Path

def main():
    print("🚀 PromptGen MCP Setup")
    print("=" * 50)
    
    # Get API keys from user
    print("\n📝 Please provide your API keys:")
    print("1. Get PromptGen API key from: https://promptgen-mcp.replit.app")
    print("2. Get GROQ API key from: https://console.groq.com/")
    print("3. Get Tavily API key from: https://tavily.com/")
    print()
    
    promptgen_key = input("Enter your PromptGen API key (pg_sk_...): ").strip()
    groq_key = input("Enter your GROQ API key (gsk_...): ").strip()
    tavily_key = input("Enter your Tavily API key (tvly_...): ").strip()
    
    # Validate keys
    if not promptgen_key.startswith("pg_sk_"):
        print("❌ PromptGen API key should start with 'pg_sk_'")
        return
    
    if not groq_key.startswith("gsk_"):
        print("❌ GROQ API key should start with 'gsk_'")
        return
    
    if not tavily_key.startswith("tvly_"):
        print("❌ Tavily API key should start with 'tvly_'")
        return
    
    # Get current directory
    current_dir = Path(__file__).parent.absolute()
    server_path = current_dir / "src" / "prompt_gen_mcp" / "server.py"
    
    if not server_path.exists():
        print(f"❌ Server file not found at: {server_path}")
        return
    
    # Create MCP configuration
    config = {
        "mcpServers": {
            "prompt-gen": {
                "command": "python",
                "args": [str(server_path)],
                "env": {
                    "PROMPTGEN_API_KEY": promptgen_key,
                    "GROQ_API_KEY": groq_key,
                    "TAVILY_API_KEY": tavily_key
                }
            }
        }
    }
    
    # Save configuration
    config_file = current_dir / "user_cursor_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Configuration saved to: {config_file}")
    print("\n📋 Next steps:")
    print("1. Copy the configuration to your Cursor MCP file:")
    print(f"   cp {config_file} ~/.cursor/mcp_servers.json")
    print("\n2. Or manually copy this to ~/.cursor/mcp_servers.json:")
    print(json.dumps(config, indent=2))
    print("\n3. Completely restart Cursor (not just reload)")
    print("4. Test with: Cmd+Shift+P → 'MCP: enhance_prompt'")
    print("\n🎉 Setup complete!")

if __name__ == "__main__":
    main() 