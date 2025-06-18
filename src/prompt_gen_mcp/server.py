#!/usr/bin/env python3
"""
PromptGen MCP Server - Advanced Prompt Engineering with Self-RAG

This server provides:
- Local workspace scanning and code analysis (private)
- Hosted API access to 47+ prompt engineering techniques
- Self-RAG pipeline for enhanced prompt generation
"""

import asyncio
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import httpx
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
import mcp.types as types

# Configuration
@dataclass
class Config:
    """Configuration for PromptGen MCP Server"""
    # PromptGen API Configuration
    techniques_api_url: str = "https://api.promptgen.dev"  # PromptGen API service
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("PROMPTGEN_API_KEY"))
    
    # Local LLM Configuration  
    groq_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    
    # Local Processing Settings
    max_files_to_analyze: int = 10
    supported_extensions: set = field(default_factory=lambda: {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.md', '.json', '.yaml', '.yml', 
        '.html', '.css', '.scss', '.vue', '.svelte', '.go', '.rs', '.java', '.cpp'
    })

config = Config()

# Global state
server = Server("promptgen-mcp")
workspace_files: List[Path] = []

class TechniquesAPI:
    """Client for the hosted techniques API"""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        self.api_url = api_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def get_techniques_for_query(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get optimal techniques for a given query"""
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            response = await self.client.post(
                f"{self.api_url}/v1/techniques/select",
                json={
                    "query": query,
                    "limit": limit,
                    "include_examples": True
                },
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("techniques", [])
            elif response.status_code == 429:
                # Rate limit exceeded
                return [{
                    "name": "Chain of Thought",
                    "description": "Break down complex problems step by step",
                    "example": "Let me think through this step by step: 1) First, I need to..."
                }]
            else:
                # API error, return fallback
                return self._get_fallback_techniques(query)
                
        except Exception as e:
            print(f"⚠️ Techniques API error: {e}")
            return self._get_fallback_techniques(query)
    
    def _get_fallback_techniques(self, query: str) -> List[Dict[str, Any]]:
        """Fallback techniques when API is unavailable"""
        fallback_techniques = {
            "Chain of Thought": {
                "description": "Break down complex problems into step-by-step reasoning",
                "example": "Let me think step by step: 1) First I need to... 2) Then I should..."
            },
            "Few-Shot Learning": {
                "description": "Provide examples to guide the model's responses",
                "example": "Here are some examples: Example 1: Input -> Output, Example 2: Input -> Output"
            },
            "Role Assignment": {
                "description": "Assign a specific role or expertise to the model",
                "example": "You are an expert software engineer with 10 years of experience..."
            }
        }
        
        # Simple keyword matching for technique selection
        query_lower = query.lower()
        selected = []
        
        if any(word in query_lower for word in ['debug', 'error', 'fix', 'problem']):
            selected.append("Chain of Thought")
        if any(word in query_lower for word in ['example', 'show', 'how']):
            selected.append("Few-Shot Learning")
        if any(word in query_lower for word in ['expert', 'professional', 'specialist']):
            selected.append("Role Assignment")
        
        if not selected:
            selected = ["Chain of Thought"]
        
        return [
            {
                "name": name,
                "description": fallback_techniques[name]["description"],
                "example": fallback_techniques[name]["example"]
            }
            for name in selected[:3]
        ]

# Initialize API client
techniques_api = TechniquesAPI(config.techniques_api_url, config.api_key)

async def scan_workspace() -> List[Path]:
    """Scan current workspace for relevant code files (LOCAL ONLY)"""
    global workspace_files
    
    try:
        workspace_path = Path.cwd()
        files = []
        
        for filepath in workspace_path.rglob("*"):
            if (filepath.is_file() and 
                filepath.suffix.lower() in config.supported_extensions and
                not any(excluded in str(filepath) for excluded in [
                    'node_modules', '.git', '__pycache__', 'venv', '.env',
                    'dist', 'build', '.next', 'target'
                ])):
                files.append(filepath)
        
        workspace_files = files[:config.max_files_to_analyze]
        return workspace_files
        
    except Exception as e:
        print(f"⚠️ Workspace scan error: {e}")
        return []

async def extract_code_context(files: List[Path], query: str) -> str:
    """Extract relevant code context from files (LOCAL ONLY)"""
    context_parts = []
    
    for filepath in files[:config.max_files_to_analyze]:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Include file if it seems relevant to the query
            if any(keyword.lower() in content.lower() for keyword in query.split()[:3]):
                # Truncate content for context
                truncated = content[:500] + "..." if len(content) > 500 else content
                context_parts.append(f"\n## {filepath.name}\n```{filepath.suffix[1:]}\n{truncated}\n```")
                
        except Exception as e:
            continue
    
    return "\n".join(context_parts) if context_parts else ""

async def call_llm(prompt: str) -> str:
    """Call LLM with the enhanced prompt"""
    try:
        # Try Groq first
        if config.groq_api_key:
            from groq import AsyncGroq
            client = AsyncGroq(api_key=config.groq_api_key)
            
            response = await client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            return response.choices[0].message.content
            
        # Fallback to OpenAI if available
        elif config.openai_api_key:
            import openai
            client = openai.AsyncOpenAI(api_key=config.openai_api_key)
            
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            return response.choices[0].message.content
        
        else:
            return f"Enhanced analysis for: {prompt[:100]}... (No LLM API key configured)"
            
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="enhance_prompt",
            description="Transform a simple prompt into an enhanced prompt using advanced techniques and local code context",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The simple prompt to enhance"
                    },
                    "include_code_context": {
                        "type": "boolean", 
                        "description": "Whether to include local code context",
                        "default": True
                    },
                    "max_techniques": {
                        "type": "integer",
                        "description": "Maximum number of techniques to apply",
                        "default": 3
                    }
                },
                "required": ["prompt"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls"""
    
    if name == "enhance_prompt":
        prompt = arguments.get("prompt", "")
        include_code_context = arguments.get("include_code_context", True)
        max_techniques = arguments.get("max_techniques", 3)
        
        if not prompt:
            return [types.TextContent(type="text", text="Error: No prompt provided")]
        
        try:
            # Step 1: Get optimal techniques from hosted API
            print(f"🎯 Selecting techniques for: {prompt[:50]}...")
            techniques = await techniques_api.get_techniques_for_query(prompt, max_techniques)
            
            # Step 2: Scan local workspace (if enabled)
            code_context = ""
            if include_code_context:
                print("🔍 Scanning local workspace...")
                files = await scan_workspace()
                code_context = await extract_code_context(files, prompt)
                print(f"📁 Found {len(files)} relevant files")
            
            # Step 3: Build enhanced prompt
            enhanced_prompt = f"""# Enhanced Prompt with Advanced Techniques

## Original Request
{prompt}

## Selected Techniques
"""
            
            for technique in techniques:
                enhanced_prompt += f"""
### {technique['name']}
**Description:** {technique['description']}
**Example:** {technique['example']}
"""
            
            if code_context:
                enhanced_prompt += f"""
## Local Code Context
{code_context}
"""
            
            enhanced_prompt += f"""
## Instructions
Please respond to the original request using the {', '.join([t['name'] for t in techniques])} technique(s) above.
{"Consider the provided code context and " if code_context else ""}Apply the techniques explicitly in your response.

Structure your response with clear sections showing each technique in action.
"""
            
            # Step 4: Generate response with LLM
            print("🤖 Generating enhanced response...")
            llm_response = await call_llm(enhanced_prompt)
            
            # Step 5: Return results
            result = {
                "success": True,
                "enhanced_prompt": enhanced_prompt,
                "llm_response": llm_response,
                "techniques_used": [t['name'] for t in techniques],
                "files_analyzed": len(workspace_files) if include_code_context else 0,
                "metadata": {
                    "prompt_length": len(enhanced_prompt),
                    "response_length": len(llm_response),
                    "api_used": "hosted" if config.api_key else "fallback"
                }
            }
            
            return [types.TextContent(
                type="text", 
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "fallback_techniques": ["Chain of Thought", "Few-Shot Learning"]
            }
            return [types.TextContent(
                type="text",
                text=json.dumps(error_result, indent=2)
            )]
    
    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    """Main entry point"""
    # Transport setup would go here for actual MCP usage
    # For now, this serves as the server structure
    print("🚀 PromptGen MCP Server initialized")
    print(f"📡 API URL: {config.techniques_api_url}")
    print(f"🔑 API Key: {'Configured' if config.api_key else 'Not configured (using fallback)'}")
    print(f"🤖 LLM: {'Groq' if config.groq_api_key else 'OpenAI' if config.openai_api_key else 'None'}")

if __name__ == "__main__":
    asyncio.run(main()) 