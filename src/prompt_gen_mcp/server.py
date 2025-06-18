#!/usr/bin/env python3
"""
PromptGen MCP Server - Advanced Prompt Engineering with Vectorized Techniques

This server provides:
- Local workspace scanning and code analysis (private)
- Qdrant Cloud access to 47+ vectorized prompt engineering techniques
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
    # PromptGen API Configuration (Required)
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("PROMPTGEN_API_KEY"))
    qdrant_url: str = "https://c1178bc4-5f80-4d4d-a6bf-c10ba4de69b9.eu-central-1-0.aws.cloud.qdrant.io:6333"
    
    # Local LLM Configuration  
    groq_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    tavily_api_key: Optional[str] = field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))
    
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

class QdrantTechniquesAPI:
    """Client for Qdrant Cloud hosted techniques"""
    
    def __init__(self, qdrant_url: str, api_key: Optional[str] = None):
        self.qdrant_url = qdrant_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)
        self.collection_name = "prompt_techniques"
    
    async def get_techniques_for_query(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get optimal techniques for a given query using vector search"""
        if not self.api_key:
            raise Exception("PROMPTGEN_API_KEY is required for technique access")
        
        try:
            # Perform vector search on Qdrant Cloud
            search_payload = {
                "vector": await self._get_query_embedding(query),
                "limit": limit,
                "with_payload": True,
                "with_vector": False
            }
            
            headers = {
                "api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            response = await self.client.post(
                f"{self.qdrant_url}/collections/{self.collection_name}/points/search",
                json=search_payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                techniques = []
                
                for result in data.get("result", []):
                    payload = result.get("payload", {})
                    techniques.append({
                        "name": payload.get("name", "Unknown Technique"),
                        "description": payload.get("description", ""),
                        "example": payload.get("example", ""),
                        "score": result.get("score", 0.0)
                    })
                
                return techniques
            elif response.status_code == 401:
                raise Exception("Invalid PROMPTGEN_API_KEY - get a valid key from promptgen.dev")
            elif response.status_code == 429:
                raise Exception("Rate limit exceeded - upgrade your plan at promptgen.dev")
            else:
                raise Exception(f"Qdrant API error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Techniques API error: {e}")
            raise e
    
    async def _get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query (simplified - in production use proper embedding model)"""
        # This is a placeholder - in production you'd use the same embedding model
        # that was used to create the vectors in Qdrant
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            embedding = model.encode(query).tolist()
            return embedding
        except ImportError:
            # Fallback to simple hash-based vector (not ideal but works for demo)
            import hashlib
            hash_obj = hashlib.md5(query.encode())
            # Convert hash to 384-dimensional vector (matching all-MiniLM-L6-v2)
            hash_int = int(hash_obj.hexdigest(), 16)
            vector = [(hash_int >> i) % 2 - 0.5 for i in range(384)]
            return vector

# Initialize API client
techniques_api = QdrantTechniquesAPI(config.qdrant_url, config.api_key)

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
        else:
            return f"Enhanced prompt ready (set GROQ_API_KEY to get LLM response):\n\n{prompt}"
            
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="enhance_prompt",
            description="Transform a simple prompt into an enhanced prompt using vectorized techniques from Qdrant Cloud and local code context",
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
        
        if not config.api_key:
            return [types.TextContent(
                type="text", 
                text="❌ Error: PROMPTGEN_API_KEY is required. Get your API key from promptgen.dev"
            )]
        
        try:
            # Step 1: Get optimal techniques from Qdrant Cloud
            print(f"🎯 Selecting vectorized techniques for: {prompt[:50]}...")
            techniques = await techniques_api.get_techniques_for_query(prompt, max_techniques)
            print(f"✅ Found {len(techniques)} optimal techniques")
            
            # Step 2: Scan local workspace (if enabled)
            code_context = ""
            if include_code_context:
                print("🔍 Scanning local workspace...")
                files = await scan_workspace()
                code_context = await extract_code_context(files, prompt)
                print(f"📁 Found {len(files)} relevant files")
            
            # Step 3: Build enhanced prompt
            enhanced_prompt = f"""# Enhanced Prompt with Vectorized Techniques

## Original Request
{prompt}

## Selected Techniques (from Qdrant Cloud)
"""
            
            for i, technique in enumerate(techniques, 1):
                score = technique.get('score', 0.0)
                enhanced_prompt += f"""
### {i}. {technique['name']} (Similarity: {score:.3f})
**Description**: {technique['description']}

**Example**: {technique['example']}
"""
            
            if code_context:
                enhanced_prompt += f"""
## Local Code Context
{code_context}
"""
            
            enhanced_prompt += f"""
## Enhanced Analysis Request
Using the techniques above and the provided code context, please provide a comprehensive response to: "{prompt}"

Apply the selected techniques systematically and reference specific code files when relevant.
"""
            
            return [types.TextContent(
                type="text",
                text=enhanced_prompt
            )]
            
        except Exception as e:
            error_msg = str(e)
            if "PROMPTGEN_API_KEY" in error_msg:
                return [types.TextContent(
                    type="text",
                    text=f"❌ API Key Error: {error_msg}\n\nGet your API key from promptgen.dev"
                )]
            else:
                return [types.TextContent(
                    type="text",
                    text=f"❌ Error: {error_msg}"
                )]
    
    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    """Main server entry point"""
    # Verify API key on startup
    if not config.api_key:
        print("⚠️  Warning: PROMPTGEN_API_KEY not set. Get your API key from promptgen.dev")
    else:
        print(f"✅ PromptGen API Key configured")
    
    print("🚀 PromptGen MCP Server starting...")
    print("📊 Features:")
    print("   🔍 Local workspace scanning (private)")
    print("   🧠 Vectorized techniques from Qdrant Cloud")
    print("   ✨ Enhanced prompt generation")
    
    # Run the server
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="promptgen-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main()) 