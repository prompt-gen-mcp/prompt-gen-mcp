#!/usr/bin/env python3
"""
MCP Self-RAG Prompt Engineering Server

An advanced Model Context Protocol (MCP) server that implements a Self-Reflective RAG pipeline
with dynamic prompt engineering technique selection using all 47+ techniques from llms.txt.

Features:
- 🎯 Self-RAG technique selection using vector similarity
- 📚 All 47+ advanced prompt engineering techniques from llms.txt  
- 🧠 Dynamic technique combination based on question analysis
- 🌐 Web search integration for enhanced context
- 🤖 Support for both GROQ and local IDE LLMs
- 📊 Intelligent complexity assessment and technique matching
"""

import os
import asyncio
import time
import uuid
import tempfile
import shutil
import json
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from typing_extensions import TypedDict
from dataclasses import dataclass, field
from enum import Enum

# MCP and FastMCP imports
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

# Core AI/ML imports (optional imports with error handling)
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    HAS_QDRANT = True
except ImportError:
    print("❌ Qdrant not available. Install with: pip install qdrant-client")
    HAS_QDRANT = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    print("❌ SentenceTransformers not available. Install with: pip install sentence-transformers")
    HAS_EMBEDDINGS = False

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    print("❌ GROQ not available. Install with: pip install groq")
    HAS_GROQ = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    print("❌ HTTPX not available. Install with: pip install httpx")
    HAS_HTTPX = False

# ----------------------
# Configuration & Enums
# ----------------------

class LLMProvider(Enum):
    GROQ = "groq"
    LOCAL = "local"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    VSCODE_LM = "vscode_lm"
    AZURE = "azure"
    GOOGLE = "google"

@dataclass
class ServerConfig:
    """Server configuration with sensible defaults"""
    groq_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    tavily_api_key: Optional[str] = field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))
    
    # LLM settings
    default_llm_provider: LLMProvider = LLMProvider.GROQ
    groq_model: str = "llama-3.3-70b-versatile"
    local_llm_endpoint: str = "http://localhost:11434/v1/chat/completions"  # Ollama default
    local_llm_model: str = "llama3.2"
    
    # Additional provider settings
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    openrouter_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    azure_api_key: Optional[str] = field(default_factory=lambda: os.getenv("AZURE_API_KEY"))
    azure_endpoint: Optional[str] = field(default_factory=lambda: os.getenv("AZURE_ENDPOINT"))
    google_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    
    # Model mappings for different providers
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    openrouter_model: str = "deepseek/deepseek-chat"
    azure_model: str = "gpt-4o"
    google_model: str = "gemini-2.0-flash-exp"
    
    # Vector database settings
    vector_db_dir: str = field(default_factory=lambda: f"./mcp_qdrant_storage_{int(time.time())}")
    embedding_model_name: str = "intfloat/multilingual-e5-large-instruct"
    
    # Text processing settings  
    chunk_size: int = 1200  # Increased for better code context
    chunk_overlap: int = 200  # Increased overlap for continuity
    max_documents_retrieved: int = 8  # More documents for better coverage
    
    # Auto-upload settings
    auto_scan_workspace: bool = True  # Automatically scan and upload workspace
    workspace_scan_extensions: set = field(default_factory=lambda: {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.md', '.txt', '.json', 
        '.yaml', '.yml', '.html', '.css', '.java', '.cpp', '.c', 
        '.h', '.rs', '.go', '.php', '.rb', '.swift', '.kt', '.sql'
    })

# Global configuration
config = ServerConfig()

# ----------------------
# FastMCP Server Setup
# ----------------------

mcp = FastMCP(
    "Self-RAG Prompt Engineering Server",
    dependencies=[
        "sentence-transformers>=2.2.2",
        "qdrant-client>=1.7.0", 
        "groq>=0.8.0",
        "httpx>=0.24.0",
        "pydantic>=2.0.0"
    ]
)

# ----------------------
# Global State Variables
# ----------------------

# Core AI components
embedding_model: Optional[Any] = None
qdrant_client: Optional[Any] = None
groq_client: Optional[Any] = None

# Advanced prompt engineering
llms_techniques: Dict[str, Dict[str, str]] = {}
technique_selection_rag_ready: bool = False

# Repository management
active_repositories: Dict[str, Dict[str, Any]] = {}
workspace_repo_id: Optional[str] = None  # Track current workspace repo
workspace_last_scan: Optional[float] = None  # Last workspace scan timestamp

# ----------------------
# Data Models
# ----------------------

class LLMConfig(BaseModel):
    """LLM configuration model"""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2000

class RepositoryInfo(BaseModel):
    """Repository information model"""
    repo_id: str
    name: str
    description: str
    file_count: int
    chunk_count: int
    indexed_at: str

class TechniqueSelection(BaseModel):
    """Selected prompt engineering techniques"""
    selected_techniques: List[str]
    reasoning: str
    complexity_level: str
    question_types: List[str]

class QueryResponse(BaseModel):
    """Query response model"""
    answer: str
    technique_selection: TechniqueSelection
    documents_used: int
    processing_time: float

# ----------------------
# Core Initialization Functions
# ----------------------

async def initialize_components():
    """Initialize all server components"""
    global embedding_model, qdrant_client, groq_client, llms_techniques
    
    try:
        # Load prompt engineering techniques
        await load_llms_txt_techniques()
        
        # Initialize embedding model if available
        if HAS_EMBEDDINGS:
            print("🔄 Loading E5 embedding model...")
            embedding_model = SentenceTransformer(config.embedding_model_name)
            print("✅ E5 embedding model loaded successfully")
        
        # Initialize vector database if available
        if HAS_QDRANT:
            qdrant_client = QdrantClient(path=config.vector_db_dir, prefer_grpc=False)
            
            # Create collections
            if not qdrant_client.collection_exists(collection_name="code_vectors"):
                qdrant_client.create_collection(
                    collection_name="code_vectors",
                    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
                )
            print("✅ Vector database initialized")
        
        # Initialize GROQ client if available
        if HAS_GROQ and config.groq_api_key:
            groq_client = Groq(api_key=config.groq_api_key)
            print("✅ GROQ client initialized")
        
        print("✅ MCP Self-RAG Server initialization complete!")
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")

async def load_llms_txt_techniques():
    """Load and parse all advanced prompt engineering techniques from llms.txt"""
    global llms_techniques
    
    try:
        llms_file = Path("llms.txt")
        if not llms_file.exists():
            print("❌ Warning: llms.txt file not found. Creating basic techniques...")
            # Create basic techniques if file doesn't exist
            llms_techniques = {
                "Chain of Thought": {
                    "definition": "Break down complex problems into step-by-step reasoning",
                    "example": "Let me think step by step: 1) First I need to... 2) Then I should..."
                },
                "Few-Shot Learning": {
                    "definition": "Provide examples to guide the model's responses",
                    "example": "Here are some examples: Example 1: ... Example 2: ..."
                },
                "Role Assignment": {
                    "definition": "Assign a specific role or persona to the model",
                    "example": "You are an expert software architect with 10+ years experience..."
                }
            }
            return llms_techniques
        
        with open(llms_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        techniques = {}
        current_technique = None
        current_definition = ""
        current_example = ""
        in_example = False
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            
            if line.startswith('**Technique:'):
                if current_technique:
                    techniques[current_technique] = {
                        'definition': current_definition.strip(),
                        'example': current_example.strip()
                    }
                current_technique = line.replace('**Technique:', '').replace('**', '').strip()
                current_definition = ""
                current_example = ""
                in_example = False
                
            elif line.startswith('**Definition:'):
                current_definition = line.replace('**Definition:**', '').strip()
                in_example = False
                
            elif line.startswith('**Example:'):
                in_example = True
                current_example = line.replace('**Example:**', '').strip()
                
            elif current_technique and line and not line.startswith('---'):
                if in_example:
                    current_example += "\n" + line
                elif not line.startswith('**'):
                    current_definition += " " + line
        
        # Add the last technique
        if current_technique:
            techniques[current_technique] = {
                'definition': current_definition.strip(),
                'example': current_example.strip()
            }
        
        llms_techniques = techniques
        print(f"✅ Loaded {len(techniques)} advanced prompt engineering techniques from llms.txt")
        return techniques
        
    except Exception as e:
        print(f"❌ Error loading llms.txt: {e}")
        return {}

# ----------------------
# LLM Abstraction Layer
# ----------------------

async def call_llm(prompt: str, llm_config: Optional[LLMConfig] = None) -> str:
    """Call LLM with abstracted interface supporting multiple providers"""
    
    # Use provided config or default
    if llm_config is None:
        llm_config = LLMConfig(
            provider=config.default_llm_provider,
            model=config.groq_model if config.default_llm_provider == LLMProvider.GROQ else config.local_llm_model,
            api_key=config.groq_api_key,
            endpoint=config.local_llm_endpoint
        )
    
    try:
        if llm_config.provider == LLMProvider.GROQ and HAS_GROQ:
            if not groq_client:
                raise ValueError("GROQ client not initialized")
            
            response = await asyncio.to_thread(
                groq_client.chat.completions.create,
                model=llm_config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens
            )
            return response.choices[0].message.content
            
        elif llm_config.provider == LLMProvider.OPENAI and HAS_HTTPX:
            # OpenAI API
            headers = {
                "Authorization": f"Bearer {llm_config.api_key or config.openai_api_key}",
                "Content-Type": "application/json"
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": llm_config.model or config.openai_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": llm_config.temperature,
                        "max_tokens": llm_config.max_tokens
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
                
        elif llm_config.provider == LLMProvider.ANTHROPIC and HAS_HTTPX:
            # Anthropic Claude API
            headers = {
                "x-api-key": llm_config.api_key or config.anthropic_api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json={
                        "model": llm_config.model or config.anthropic_model,
                        "max_tokens": llm_config.max_tokens,
                        "messages": [{"role": "user", "content": prompt}]
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                return response.json()["content"][0]["text"]
                
        elif llm_config.provider == LLMProvider.OPENROUTER and HAS_HTTPX:
            # OpenRouter API (OpenAI-compatible)
            headers = {
                "Authorization": f"Bearer {llm_config.api_key or config.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/your-repo",  # Optional
                "X-Title": "MCP Self-RAG Server"  # Optional
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": llm_config.model or config.openrouter_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": llm_config.temperature,
                        "max_tokens": llm_config.max_tokens
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
                
        elif llm_config.provider == LLMProvider.AZURE and HAS_HTTPX:
            # Azure OpenAI API
            headers = {
                "api-key": llm_config.api_key or config.azure_api_key,
                "Content-Type": "application/json"
            }
            endpoint = llm_config.endpoint or config.azure_endpoint
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{endpoint}/openai/deployments/{llm_config.model or config.azure_model}/chat/completions?api-version=2024-02-15-preview",
                    headers=headers,
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": llm_config.temperature,
                        "max_tokens": llm_config.max_tokens
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
                
        elif llm_config.provider == LLMProvider.GOOGLE and HAS_HTTPX:
            # Google Gemini API
            headers = {
                "Content-Type": "application/json"
            }
            api_key = llm_config.api_key or config.google_api_key
            model = llm_config.model or config.google_model
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                    headers=headers,
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": llm_config.temperature,
                            "maxOutputTokens": llm_config.max_tokens
                        }
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
                
        elif llm_config.provider == LLMProvider.VSCODE_LM:
            # VS Code Language Model API (placeholder - would need VS Code extension context)
            print("⚠️ VS Code LM API requires VS Code extension context")
            return "VS Code LM API integration requires running within VS Code extension context"
            
        elif llm_config.provider == LLMProvider.LOCAL and HAS_HTTPX:
            # Call local LLM endpoint (e.g., Ollama, LM Studio, etc.)
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    llm_config.endpoint,
                    json={
                        "model": llm_config.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": llm_config.temperature,
                        "max_tokens": llm_config.max_tokens
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
        
        else:
            # Fallback to simple response
            return f"LLM response for: {prompt[:100]}... (Mock response - LLM not available)"
            
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return f"Error calling LLM: {str(e)}"

# ----------------------
# Advanced Technique Selection
# ----------------------

async def select_optimal_techniques(question: str, context: str = "") -> TechniqueSelection:
    """Select optimal prompt engineering techniques for the given question"""
    
    # Analyze question type and complexity
    question_lower = question.lower()
    
    # Simple pattern matching for question types
    question_types = []
    if any(word in question_lower for word in ['debug', 'error', 'fix', 'bug']):
        question_types.append('debugging')
    if any(word in question_lower for word in ['optimize', 'performance', 'speed']):
        question_types.append('optimization')  
    if any(word in question_lower for word in ['explain', 'how', 'what', 'why']):
        question_types.append('explanation')
    if any(word in question_lower for word in ['compare', 'difference', 'versus']):
        question_types.append('comparison')
    
    if not question_types:
        question_types = ['general']
    
    # Determine complexity
    complexity = 'high' if len(question.split()) > 20 else 'medium' if len(question.split()) > 10 else 'low'
    
    # Select techniques based on analysis
    selected_techniques = ['Chain of Thought']  # Always include CoT
    
    if 'debugging' in question_types:
        selected_techniques.extend(['Few-Shot Learning', 'Step-Back Prompting'])
    if 'optimization' in question_types:
        selected_techniques.extend(['Plan and Solve', 'Tree of Thought'])
    if 'explanation' in question_types:
        selected_techniques.extend(['Few-Shot Learning', 'Role Assignment'])
    if complexity == 'high':
        selected_techniques.extend(['Self-Ask', 'Chain of Verification'])
    
    # Remove duplicates while preserving order
    selected_techniques = list(dict.fromkeys(selected_techniques))
    
    # Limit to available techniques
    available_techniques = list(llms_techniques.keys())
    selected_techniques = [t for t in selected_techniques if t in available_techniques]
    
    # Fallback if no techniques available
    if not selected_techniques and available_techniques:
        selected_techniques = available_techniques[:3]
    elif not selected_techniques:
        selected_techniques = ['Chain of Thought', 'Few-Shot Learning', 'Role Assignment']
    
    reasoning = f"Selected {len(selected_techniques)} techniques based on question types: {', '.join(question_types)} with {complexity} complexity"
    
    return TechniqueSelection(
        selected_techniques=selected_techniques,
        reasoning=reasoning,
        complexity_level=complexity,
        question_types=question_types
    )

# ----------------------
# Document Processing & Vector Search
# ----------------------

async def process_uploaded_files(temp_path: Path, repo_id: str) -> Dict[str, Any]:
    """Process uploaded files and index them in vector database"""
    try:
        documents = []
        processed_files = 0
        
        print(f"🔄 Processing files from: {temp_path}")
        
        # Supported file extensions
        supported_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.md', '.txt', '.json', 
                               '.yaml', '.yml', '.html', '.css', '.java', '.cpp', '.c', 
                               '.h', '.rs', '.go', '.php', '.rb', '.swift', '.kt', '.sql'}
        
        for filepath in temp_path.rglob("*"):
            if filepath.is_file() and not filepath.name.startswith('.'):
                try:
                    if filepath.suffix.lower() in supported_extensions:
                        text = filepath.read_text(encoding='utf-8', errors='ignore')
                        if text.strip():
                            # Smart chunking for code files
                            chunks = []
                            chunk_size = config.chunk_size
                            overlap = config.chunk_overlap
                            
                            # For code files, try to preserve logical structure
                            if filepath.suffix.lower() in {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.rs', '.go'}:
                                # Split by functions/classes first, then by lines if still too big
                                lines = text.split('\n')
                                current_chunk = []
                                current_size = 0
                                
                                for line in lines:
                                    line_size = len(line) + 1  # +1 for newline
                                    
                                    # If adding this line exceeds chunk size and we have content
                                    if current_size + line_size > chunk_size and current_chunk:
                                        chunk_text = '\n'.join(current_chunk)
                                        if chunk_text.strip():
                                            chunks.append(chunk_text)
                                        
                                        # Start new chunk with overlap
                                        overlap_lines = max(0, len(current_chunk) - overlap // 50)  # Rough line estimation
                                        current_chunk = current_chunk[overlap_lines:] + [line]
                                        current_size = sum(len(l) + 1 for l in current_chunk)
                                    else:
                                        current_chunk.append(line)
                                        current_size += line_size
                                
                                # Add final chunk
                                if current_chunk:
                                    chunk_text = '\n'.join(current_chunk)
                                    if chunk_text.strip():
                                        chunks.append(chunk_text)
                            
                            else:
                                # Regular chunking for non-code files  
                                for i in range(0, len(text), chunk_size - overlap):
                                    chunk = text[i:i + chunk_size]
                                    if chunk.strip():
                                        chunks.append(chunk)
                            
                            # Create document entries
                            for i, chunk in enumerate(chunks):
                                documents.append({
                                    'id': str(uuid.uuid4()),
                                    'text': chunk,
                                    'metadata': {
                                        'file_path': str(filepath.relative_to(temp_path)),
                                        'chunk_index': i,
                                        'repo_id': repo_id,
                                        'file_type': filepath.suffix,
                                        'total_chunks': len(chunks)
                                    }
                                })
                            processed_files += 1
                except Exception as e:
                    print(f"❌ Error processing {filepath}: {e}")
                    continue
        
        print(f"📊 Processed {processed_files} files into {len(documents)} chunks")
        
        if not documents:
            return {"error": "No processable files found", "files_processed": 0, "chunks": 0}
        
        # Embed document chunks if embedding model available
        if embedding_model and HAS_EMBEDDINGS:
            print(f"🔄 Generating embeddings for {len(documents)} chunks...")
            
            # Prepare texts for embedding
            doc_texts = []
            for doc in documents:
                # Enhanced format for E5 embedding model with file context
                file_path = doc['metadata']['file_path']
                file_type = doc['metadata']['file_type']
                text = doc['text']
                
                # Create context-aware embedding text
                if file_type in {'.py', '.js', '.ts', '.jsx', '.tsx'}:
                    instruct_text = f"passage: {file_path} - {text}"
                else:
                    instruct_text = f"passage: {text}"
                
                doc_texts.append(instruct_text)
            
            # Generate embeddings in batches
            batch_size = 50
            all_embeddings = []
            
            for i in range(0, len(doc_texts), batch_size):
                batch = doc_texts[i:i + batch_size]
                batch_embeddings = embedding_model.encode(batch, convert_to_tensor=True)
                all_embeddings.extend(batch_embeddings.cpu().numpy().tolist())
            
            print(f"✅ Generated {len(all_embeddings)} embeddings")
            
            # Store in Qdrant if available
            if qdrant_client and HAS_QDRANT:
                points = []
                for doc, embedding in zip(documents, all_embeddings):
                    point = PointStruct(
                        id=doc['id'],
                        vector=embedding,
                        payload={
                            **doc['metadata'],
                            'text': doc['text']  # Store text in payload for retrieval
                        }
                    )
                    points.append(point)
                
                # Insert in batches
                batch_size = 100
                for i in range(0, len(points), batch_size):
                    batch_points = points[i:i + batch_size]
                    qdrant_client.upsert(
                        collection_name="code_vectors",
                        points=batch_points
                    )
                
                print(f"✅ Stored {len(points)} vectors in Qdrant")
        
        return {
            "success": True,
            "files_processed": processed_files,
            "chunks_created": len(documents),
            "repo_id": repo_id
        }
        
    except Exception as e:
        print(f"❌ Error processing files: {e}")
        return {"error": str(e), "files_processed": 0, "chunks": 0}

# ----------------------
# Auto Workspace Management
# ----------------------

async def get_workspace_hash() -> str:
    """Generate a hash of the current workspace to detect changes"""
    import hashlib
    
    try:
        workspace_path = Path.cwd()
        hash_content = []
        file_count = 0
        
        # Scan all relevant files in workspace
        for filepath in workspace_path.rglob("*"):
            if (filepath.is_file() and 
                not filepath.name.startswith('.') and
                filepath.suffix.lower() in config.workspace_scan_extensions and
                not any(ignore in str(filepath) for ignore in ['.git', 'node_modules', '__pycache__', '.venv', 'venv', 'mcp_qdrant_storage'])):
                
                try:
                    # Include file path and file size (more reliable than mtime)
                    stat = filepath.stat()
                    rel_path = filepath.relative_to(workspace_path)
                    hash_content.append(f"{rel_path}:{stat.st_size}:{stat.st_mtime}")
                    file_count += 1
                    
                    # Limit to prevent excessive computation
                    if file_count > 1000:
                        break
                        
                except Exception:
                    continue
        
        # Create hash from all file paths, sizes, and modification times
        content_str = '\n'.join(sorted(hash_content))
        workspace_hash = hashlib.md5(content_str.encode()).hexdigest()
        
        print(f"📊 Workspace hash: {workspace_hash[:8]}... (from {file_count} files)")
        return workspace_hash
        
    except Exception as e:
        print(f"❌ Error generating workspace hash: {e}")
        return str(time.time())  # Fallback to timestamp

async def scan_and_upload_workspace() -> Optional[str]:
    """Automatically scan and upload the current workspace"""
    global workspace_repo_id, workspace_last_scan
    
    try:
        if not config.auto_scan_workspace:
            return workspace_repo_id
        
        workspace_path = Path.cwd()
        current_hash = await get_workspace_hash()
        current_time = time.time()
        
        # Check if workspace already exists and hasn't changed
        if workspace_repo_id and workspace_repo_id in active_repositories:
            repo_info = active_repositories[workspace_repo_id]
            stored_hash = repo_info.get('workspace_hash', '')
            last_scan_time = workspace_last_scan or 0
            
            # If hash matches and we scanned recently, skip re-upload
            if (stored_hash == current_hash and 
                current_time - last_scan_time < 300):  # 5 minutes cache
                print(f"✅ Using cached workspace: {workspace_repo_id} (no changes detected)")
                return workspace_repo_id
            
            # If hash is different, we need to re-upload
            if stored_hash != current_hash:
                print(f"🔄 Workspace changes detected, re-uploading...")
                # Clean up old vectors for this workspace
                if qdrant_client and HAS_QDRANT:
                    try:
                        # Delete old vectors by filtering on repo_id
                        qdrant_client.delete(
                            collection_name="code_vectors",
                            points_selector={"filter": {"must": [{"key": "repo_id", "match": {"value": workspace_repo_id}}]}}
                        )
                        print(f"🗑️ Cleaned up old vectors for workspace")
                    except Exception as e:
                        print(f"⚠️ Could not clean up old vectors: {e}")
            else:
                print(f"✅ Using existing workspace: {workspace_repo_id} (cache expired)")
                workspace_last_scan = current_time
                return workspace_repo_id
        
        # Check if we need to scan (first time or changes detected)
        should_scan = (
            workspace_repo_id is None or 
            workspace_last_scan is None or
            current_time - workspace_last_scan > 300  # Re-scan every 5 minutes max
        )
        
        if not should_scan and workspace_repo_id:
            print(f"✅ Using existing workspace: {workspace_repo_id}")
            return workspace_repo_id
        
        print(f"🔄 Auto-scanning workspace: {workspace_path}")
        
        # Create temporary directory with workspace files
        temp_dir = Path(tempfile.mkdtemp(prefix="workspace_scan_"))
        try:
            # Copy relevant files to temp directory
            files_copied = 0
            for filepath in workspace_path.rglob("*"):
                if (filepath.is_file() and 
                    not filepath.name.startswith('.') and
                    filepath.suffix.lower() in config.workspace_scan_extensions and
                    not any(ignore in str(filepath) for ignore in ['.git', 'node_modules', '__pycache__', '.venv', 'venv', 'mcp_qdrant_storage'])):
                    
                    try:
                        # Create relative path in temp directory
                        rel_path = filepath.relative_to(workspace_path)
                        dest_path = temp_dir / rel_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Copy file
                        import shutil
                        shutil.copy2(filepath, dest_path)
                        files_copied += 1
                        
                        if files_copied > 1000:  # Limit to prevent huge uploads
                            break
                            
                    except Exception as e:
                        print(f"⚠️ Could not copy {filepath}: {e}")
                        continue
            
            if files_copied == 0:
                print("⚠️ No relevant files found in workspace")
                return workspace_repo_id
            
            print(f"📁 Copied {files_copied} files for indexing")
            
            # Generate repo ID for workspace
            if workspace_repo_id is None:
                workspace_repo_id = f"workspace_{int(current_time)}"
            
            # Process and index files
            processing_result = await process_uploaded_files(temp_dir, workspace_repo_id)
            
            if processing_result.get("success"):
                # Update or create repository info
                active_repositories[workspace_repo_id] = {
                    "name": f"Workspace ({workspace_path.name})",
                    "description": f"Auto-uploaded workspace from {workspace_path}",
                    "temp_path": str(temp_dir),
                    "files_processed": processing_result["files_processed"],
                    "chunks_created": processing_result["chunks_created"],
                    "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "workspace_hash": current_hash,
                    "auto_uploaded": True
                }
                
                workspace_last_scan = current_time
                print(f"✅ Workspace auto-uploaded successfully! ({processing_result['files_processed']} files, {processing_result['chunks_created']} chunks)")
                
                return workspace_repo_id
            else:
                print(f"❌ Workspace upload failed: {processing_result.get('error', 'Unknown error')}")
                return None
                
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
        
    except Exception as e:
        print(f"❌ Auto workspace scan failed: {e}")
        return workspace_repo_id

async def search_documents(query: str, repo_id: Optional[str] = None, limit: int = 8) -> List[Dict[str, Any]]:
    """Search for documents using vector similarity with fallback strategies"""
    try:
        if not qdrant_client or not embedding_model:
            print("❌ Vector database or embedding model not initialized")
            return []
        
        # Generate query embedding with instruction prefix for code
        instruction_prefix = "query: "
        if any(keyword in query.lower() for keyword in ['code', 'function', 'implement', 'optimize', 'react', 'component']):
            instruction_prefix = "query: Find relevant code snippets for: "
        
        query_with_instruction = f"{instruction_prefix}{query}"
        query_embedding = embedding_model.encode([query_with_instruction])[0].tolist()
        
        documents = []
        
        # Try multiple search strategies
        search_strategies = []
        
        # The main collection where all documents are stored
        search_strategies.append("code_vectors")
        
        # Legacy collection names for backward compatibility
        if repo_id:
            search_strategies.append(f"repo_{repo_id}")
        search_strategies.append("documents")
        
        for collection_name in search_strategies:
            try:
                # Check if collection exists
                collections = qdrant_client.get_collections()
                collection_names = [col.name for col in collections.collections]
                
                if collection_name not in collection_names:
                    print(f"⚠️ Collection '{collection_name}' does not exist, skipping")
                    continue
                
                # Perform vector search with lower threshold
                search_params = {
                    "collection_name": collection_name,
                    "query_vector": query_embedding,
                    "limit": limit,
                    "with_payload": True,
                    "with_vectors": False,
                    "score_threshold": 0.1  # Lower threshold to include more results
                }
                
                # Add repo_id filter for main collection
                if collection_name == "code_vectors" and repo_id:
                    from qdrant_client.models import Filter, FieldCondition, MatchValue
                    search_params["query_filter"] = Filter(
                        must=[
                            FieldCondition(
                                key="repo_id",
                                match=MatchValue(value=repo_id)
                            )
                        ]
                    )
                
                search_results = qdrant_client.search(**search_params)
                
                # Convert results to dictionaries
                for result in search_results:
                    doc_dict = {
                        'text': result.payload.get('text', ''),
                        'file_path': result.payload.get('file_path', ''),
                        'chunk_index': result.payload.get('chunk_index', 0),
                        'score': float(result.score),
                        'metadata': result.payload,
                        'collection': collection_name
                    }
                    
                    # Avoid duplicates
                    if not any(doc['text'] == doc_dict['text'] for doc in documents):
                        documents.append(doc_dict)
                
                # If we found good results, we can break
                if len(documents) >= limit // 2:
                    break
                    
            except Exception as e:
                print(f"⚠️ Search failed for collection '{collection_name}': {e}")
                continue
        
        # Sort by score and take top results
        documents.sort(key=lambda x: x['score'], reverse=True)
        documents = documents[:limit]
        
        print(f"🔍 Found {len(documents)} relevant documents for query: '{query[:50]}...'")
        
        # Log some debug info
        if documents:
            print(f"📄 Top result: {documents[0]['file_path']} (score: {documents[0]['score']:.3f})")
        
        return documents
        
    except Exception as e:
        print(f"❌ Document search error: {e}")
        return []

# ----------------------
# Web Search Integration  
# ----------------------

async def web_search_tavily(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Perform web search using Tavily API if available"""
    try:
        if not config.tavily_api_key or not HAS_HTTPX:
            return []
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": config.tavily_api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                    "include_answer": True,
                    "include_raw_content": False
                },
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                web_docs = []
                
                for result in data.get("results", []):
                    web_docs.append({
                        'title': result.get('title', 'No Title'),
                        'url': result.get('url', ''),
                        'content': result.get('content', ''),
                        'score': result.get('score', 0.0)
                    })
                
                print(f"🌐 Found {len(web_docs)} web search results")
                return web_docs
            else:
                print(f"❌ Tavily API error: {response.status_code}")
                return []
                
    except Exception as e:
        print(f"❌ Web search error: {e}")
        return []

# ----------------------
# Self-RAG Pipeline Functions
# ----------------------

async def grade_document_relevance(document: str, question: str) -> Dict[str, Any]:
    """Grade if a document is relevant to the question using semantic analysis"""
    
    try:
        # Use semantic similarity instead of simple keyword matching
        if not embedding_model:
            return {"relevant": True, "reasoning": "No embedding model available, defaulting to relevant"}
        
        # Create embeddings for question and document
        question_embedding = embedding_model.encode([f"query: {question}"])[0]
        document_embedding = embedding_model.encode([f"passage: {document[:500]}"])[0]  # First 500 chars
        
        # Calculate cosine similarity
        from numpy import dot
        from numpy.linalg import norm
        
        similarity = dot(question_embedding, document_embedding) / (norm(question_embedding) * norm(document_embedding))
        
        # Use a lower threshold for relevance to be more inclusive
        threshold = 0.3
        is_relevant = similarity > threshold
        
        # Additional checks for code-related content
        question_lower = question.lower()
        document_lower = document.lower()
        
        # Enhanced keyword lists for different types of content
        code_keywords = ['function', 'const', 'var', 'let', 'class', 'import', 'export', 'component', 'jsx', 'tsx', 'react', 'vue', 'angular']
        tech_keywords = ['optimize', 'performance', 'bug', 'error', 'fix', 'debug', 'implement', 'refactor']
        
        # Boost relevance if both question and document contain code/tech keywords
        question_has_code = any(keyword in question_lower for keyword in code_keywords + tech_keywords)
        document_has_code = any(keyword in document_lower for keyword in code_keywords)
        
        if question_has_code and document_has_code:
            is_relevant = True
            reasoning = f"Semantic similarity: {similarity:.3f}, Code content detected in both"
        elif similarity > 0.5:  # High semantic similarity
            is_relevant = True
            reasoning = f"High semantic similarity: {similarity:.3f}"
        else:
            reasoning = f"Semantic similarity: {similarity:.3f}, threshold: {threshold}"
        
        return {
            "relevant": is_relevant,
            "reasoning": reasoning,
            "similarity_score": float(similarity)
        }
        
    except Exception as e:
        print(f"❌ Document grading error: {e}")
        return {"relevant": True, "reasoning": "Error in grading, defaulting to relevant"}

async def self_rag_pipeline(query: str, repo_id: Optional[str] = None, llm_config: Optional[LLMConfig] = None) -> Dict[str, Any]:
    """Execute the complete Self-RAG pipeline"""
    
    pipeline_start = time.time()
    results = {
        "query": query,
        "repo_id": repo_id,
        "steps": [],
        "documents_used": 0,
        "web_results_used": 0,
        "technique_selection": {},
        "final_answer": "",
        "processing_time": 0
    }
    
    try:
        # Step 1: Document Retrieval  
        print("🔍 STEP 1: Document Retrieval")
        documents = await search_documents(query, repo_id, limit=12)  # More documents initially
        results["steps"].append(f"Document retrieval completed - found {len(documents)} documents")
        
        # Step 2: Document Relevance Grading
        print("📊 STEP 2: Document Relevance Grading")
        relevant_docs = []
        for doc in documents:
            grade = await grade_document_relevance(doc['text'], query)
            if grade["relevant"]:
                relevant_docs.append(doc)
                
        results["documents_used"] = len(relevant_docs)
        results["steps"].append(f"Filtered to {len(relevant_docs)} relevant documents")
        
        # Step 3: Web Search if needed
        web_results = []
        if len(relevant_docs) < 3:  # Need more context
            print("🌐 STEP 3: Web Search for Additional Context")
            web_results = await web_search_tavily(query)
            results["web_results_used"] = len(web_results)
            results["steps"].append(f"Added {len(web_results)} web search results")
        else:
            results["steps"].append("Sufficient code context found, skipping web search")
        
        # Step 4: Technique Selection
        print("🎯 STEP 4: Optimal Technique Selection")
        all_context = "\n\n".join([doc['text'] for doc in relevant_docs])
        web_context = "\n\n".join([f"Web: {result['content']}" for result in web_results])
        full_context = f"{all_context}\n\n{web_context}".strip()
        
        technique_selection = await select_optimal_techniques(query, full_context)
        results["technique_selection"] = technique_selection.dict()
        results["steps"].append(f"Selected techniques: {', '.join(technique_selection.selected_techniques)}")
        
        # Step 5: Advanced Generation
        print("✨ STEP 5: Advanced Answer Generation")
        
        # Build comprehensive prompt
        prompt_parts = [
            f"**ADVANCED SELF-RAG RESPONSE**",
            f"",
            f"**QUESTION:** {query}",
            f"",
        ]
        
        if relevant_docs:
            prompt_parts.extend([
                f"**RELEVANT CODE/DOCUMENTS:**",
                f"{'='*50}",
            ])
            for i, doc in enumerate(relevant_docs[:4]):  # Top 4 docs
                file_path = doc['file_path']
                prompt_parts.extend([
                    f"Document {i+1} ({file_path}):",
                    f"{doc['text'][:600]}...",
                    f"",
                ])
        
        if web_results:
            prompt_parts.extend([
                f"**WEB SEARCH RESULTS:**",
                f"{'='*50}",
            ])
            for i, result in enumerate(web_results[:2]):  # Top 2 web results
                prompt_parts.extend([
                    f"Web Result {i+1}: {result['title']}",
                    f"{result['content'][:400]}...",
                    f"",
                ])
        
        # Get the actual technique content from llms.txt
        technique_content = []
        for technique_name in technique_selection.selected_techniques:
            if technique_name in llms_techniques:
                technique_info = llms_techniques[technique_name]
                technique_content.append(f"**{technique_name}:**")
                technique_content.append(f"Definition: {technique_info.get('definition', 'No definition available')}")
                technique_content.append(f"Example: {technique_info.get('example', 'No example available')}")
                technique_content.append("")
        
        prompt_parts.extend([
            f"**SELECTED PROMPT ENGINEERING TECHNIQUES:**",
            f"Selected: {', '.join(technique_selection.selected_techniques)}",
            f"Reasoning: {technique_selection.reasoning}",
            f"Complexity: {technique_selection.complexity_level}",
            f"",
            f"**TECHNIQUE DETAILS:**",
        ])
        
        prompt_parts.extend(technique_content)
        
        # Add relevant code context from workspace/documents  
        if relevant_docs:
            prompt_parts.extend([
                f"**RELEVANT CODE CONTEXT:**",
                f"Based on your current workspace, here are the most relevant code files:",
                f""
            ])
            
            # Show more relevant documents with better formatting
            for i, doc in enumerate(relevant_docs[:6]):  # Show top 6 most relevant
                file_path = doc['file_path']
                score = doc['score']
                content = doc['text']
                
                # Truncate very long content but preserve more context
                if len(content) > 1500:
                    content_preview = content[:1500] + f"\n... (truncated, showing first 1500 chars)"
                else:
                    content_preview = content
                
                prompt_parts.extend([
                    f"**📄 File: {file_path}** (relevance: {score:.3f})",
                    f"```",
                    content_preview,
                    f"```",
                    f""
                ])
        else:
            prompt_parts.extend([
                f"**⚠️ CODE CONTEXT:**",
                f"No relevant code files found in workspace for this query.",
                f"Providing general guidance without specific code context.",
                f""
            ])
        
        # Generate the actual enhanced prompt based on selected techniques
        enhanced_instructions = []
        
        for technique_name in technique_selection.selected_techniques:
            if technique_name == "Plan and Solve":
                enhanced_instructions.extend([
                    f"**STEP 1 - CREATE A PLAN:**",
                    f"First, analyze the provided code context and create a detailed plan. Your plan should include:",
                    f"• Analysis of the current code structure and potential issues",
                    f"• Specific optimization strategies based on the actual code shown above",
                    f"• Implementation steps referencing the exact files and functions",
                    f"• Expected performance improvements for each change",
                    f"",
                    f"**STEP 2 - EXECUTE THE PLAN:**",
                    f"Now follow your plan step-by-step to provide the optimization solution.",
                    f"For each step, provide specific code changes with file paths and line numbers.",
                    f"Include before/after code examples where applicable.",
                    f"Explain why each change improves performance."
                ])
            elif technique_name == "Step-Back Prompting":
                enhanced_instructions.extend([
                    f"**STEP BACK - BROADER PRINCIPLES:**",
                    f"Before addressing the specific question, first consider:",
                    f"• What are the fundamental principles of code optimization?",
                    f"• What are the key performance bottlenecks in applications?",
                    f"• What are the general best practices for the technology involved?",
                    f"",
                    f"**APPLY TO SPECIFIC CASE:**",
                    f"Now apply these broader principles to answer the specific question."
                ])
            elif technique_name == "Chain of Thought" or "Chain of Thought" in technique_name:
                enhanced_instructions.extend([
                    f"**REASONING PROCESS:**",
                    f"Think through this step-by-step, showing your reasoning:",
                    f"• What is the current situation?",
                    f"• What are the potential issues or improvements?",
                    f"• What are the possible solutions?",
                    f"• What is the best approach and why?",
                    f"• How would you implement this solution?"
                ])
            elif technique_name == "Few-Shot Learning":
                enhanced_instructions.extend([
                    f"**PROVIDE EXAMPLES:**",
                    f"Include concrete examples in your response:",
                    f"• Show before/after code examples",
                    f"• Provide specific implementation examples",
                    f"• Include real-world scenarios where this applies"
                ])
            elif "Role" in technique_name or "Role Assignment" in technique_name:
                enhanced_instructions.extend([
                    f"**EXPERT PERSPECTIVE:**",
                    f"Respond as a senior software engineer with expertise in the relevant technology.",
                    f"Use professional terminology and provide industry-standard solutions.",
                    f"Include best practices and common pitfalls to avoid."
                ])
        
        # If no specific technique instructions, provide general enhanced guidance
        if not enhanced_instructions:
            enhanced_instructions = [
                f"**ENHANCED APPROACH:**",
                f"Provide a comprehensive, well-structured response that:",
                f"• Addresses all aspects of the question thoroughly",
                f"• Uses clear, logical organization",
                f"• Includes specific, actionable advice",
                f"• Explains the reasoning behind recommendations"
            ]
        
        prompt_parts.extend([
            f"**ENHANCED PROMPT:**",
            f"Using the {', '.join(technique_selection.selected_techniques)} technique(s), answer this question:",
            f"'{query}'",
            f"",
        ])
        
        prompt_parts.extend(enhanced_instructions)
        
        prompt_parts.extend([
            f"",
            f"**BEGIN YOUR RESPONSE:**"
        ])
        
        final_prompt = "\n".join(prompt_parts)
        
        # Return the enhanced prompt instead of generating a response
        results["enhanced_prompt"] = final_prompt
        results["final_answer"] = final_prompt  # For compatibility
        results["steps"].append("Generated enhanced prompt with selected techniques")
        
        # Calculate processing time
        results["processing_time"] = round(time.time() - pipeline_start, 2)
        
        print(f"✅ Self-RAG pipeline completed in {results['processing_time']}s")
        return results
        
    except Exception as e:
        results["error"] = str(e)
        results["processing_time"] = round(time.time() - pipeline_start, 2)
        print(f"❌ Self-RAG pipeline error: {e}")
        return results

# ----------------------
# Core MCP Tools (Updated)
# ----------------------

@mcp.tool()
async def upload_repository(
    zip_file_path: str = Field(description="Path to the ZIP file containing the repository"),
    repo_name: str = Field(description="Name for this repository"),
    description: str = Field(description="Description of the repository", default="")
) -> Dict[str, Any]:
    """Upload and index a repository from a ZIP file"""
    
    await ensure_initialized()
    try:
        repo_id = str(uuid.uuid4())
        
        # Verify ZIP file exists
        zip_path = Path(zip_file_path)
        if not zip_path.exists():
            return {"error": f"ZIP file not found: {zip_file_path}"}
        
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix=f"repo_{repo_id}_"))
        
        # Extract ZIP file
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                extract_dir = temp_dir / "extracted"
                zip_ref.extractall(extract_dir)
                print(f"✅ Extracted ZIP to: {extract_dir}")
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return {"error": f"Failed to extract ZIP file: {str(e)}"}
        
        # Process and index files
        processing_result = await process_uploaded_files(extract_dir, repo_id)
        
        if processing_result.get("success"):
            # Store repository info
            active_repositories[repo_id] = {
                "name": repo_name,
                "description": description,
                "temp_path": str(temp_dir),
                "files_processed": processing_result["files_processed"],
                "chunks_created": processing_result["chunks_created"],
                "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return {
                "success": True,
                "repo_id": repo_id,
                "name": repo_name,
                "files_processed": processing_result["files_processed"],
                "chunks_created": processing_result["chunks_created"],
                "message": "Repository uploaded and indexed successfully"
            }
        else:
            # Clean up on failure
            shutil.rmtree(temp_dir, ignore_errors=True)
            return {
                "success": False,
                "error": processing_result.get("error", "Unknown processing error")
            }
            
    except Exception as e:
        return {"success": False, "error": f"Upload failed: {str(e)}"}

@mcp.tool()
async def list_repositories() -> Dict[str, Any]:
    """List all uploaded and indexed repositories"""
    
    repos_info = []
    for repo_id, info in active_repositories.items():
        repos_info.append({
            "repo_id": repo_id,
            "name": info["name"],
            "description": info["description"],
            "files_processed": info["files_processed"],
            "chunks_created": info["chunks_created"],
            "uploaded_at": info["uploaded_at"]
        })
    
    return {
        "total_repositories": len(active_repositories),
        "repositories": repos_info
    }

@mcp.tool()
async def delete_repository(
    repo_id: str = Field(description="Repository ID to delete")
) -> Dict[str, Any]:
    """Delete a repository and clean up its data"""
    
    try:
        if repo_id not in active_repositories:
            return {"success": False, "error": "Repository not found"}
        
        repo_info = active_repositories[repo_id]
        
        # Clean up temporary directory
        temp_path = Path(repo_info["temp_path"])
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)
        
        # TODO: Remove from Qdrant (would need to filter by repo_id)
        # This is a limitation of the current implementation
        
        # Remove from active repositories
        del active_repositories[repo_id]
        
        return {
            "success": True,
            "message": f"Repository '{repo_info['name']}' deleted successfully"
        }
        
    except Exception as e:
        return {"success": False, "error": f"Failed to delete repository: {str(e)}"}

@mcp.tool()
async def get_server_status() -> Dict[str, Any]:
    """Get comprehensive server status and configuration"""
    await ensure_initialized()
    return {
        "status": "running",
        "version": "1.0.0",
        "components": {
            "embedding_model": embedding_model is not None,
            "vector_database": qdrant_client is not None,
            "groq_client": groq_client is not None,
            "techniques_loaded": len(llms_techniques),
        },
        "configuration": {
            "default_llm_provider": config.default_llm_provider.value,
            "embedding_model": config.embedding_model_name,
            "chunk_size": config.chunk_size,
            "max_documents": config.max_documents_retrieved
        },
        "active_repositories": len(active_repositories),
        "capabilities": {
            "qdrant": HAS_QDRANT,
            "embeddings": HAS_EMBEDDINGS,
            "groq": HAS_GROQ,
            "httpx": HAS_HTTPX
        }
    }

@mcp.tool()
async def list_available_techniques() -> Dict[str, Any]:
    """List all available prompt engineering techniques from llms.txt"""
    await ensure_initialized()
    if not llms_techniques:
        return {
            "error": "No techniques loaded. Ensure llms.txt is available and server is initialized.",
            "techniques": []
        }
    
    techniques_info = []
    for name, details in llms_techniques.items():
        techniques_info.append({
            "name": name,
            "definition": details["definition"][:200] + "..." if len(details["definition"]) > 200 else details["definition"],
            "has_example": bool(details.get("example", "").strip())
        })
    
    return {
        "total_techniques": len(llms_techniques),
        "techniques": techniques_info
    }

@mcp.tool()
async def configure_llm_provider(
    provider: str = Field(description="LLM provider: 'groq', 'local', 'openai', 'anthropic', 'openrouter', 'azure', 'google', or 'vscode_lm'"),
    model: str = Field(description="Model name to use"),
    api_key: Optional[str] = Field(description="API key if required", default=None),
    endpoint: Optional[str] = Field(description="Endpoint URL for local/azure providers", default=None)
) -> Dict[str, str]:
    """Configure LLM provider settings for the Self-RAG system"""
    
    try:
        # Validate provider
        valid_providers = [p.value for p in LLMProvider]
        if provider not in valid_providers:
            return {
                "error": f"Invalid provider. Must be one of: {', '.join(valid_providers)}",
                "available_providers": valid_providers
            }
        
        # Update global config
        config.default_llm_provider = LLMProvider(provider)
        
        # Set model based on provider
        if provider == "groq":
            config.groq_model = model
            if api_key:
                config.groq_api_key = api_key
        elif provider == "openai":
            config.openai_model = model
            if api_key:
                config.openai_api_key = api_key
        elif provider == "anthropic":
            config.anthropic_model = model
            if api_key:
                config.anthropic_api_key = api_key
        elif provider == "openrouter":
            config.openrouter_model = model
            if api_key:
                config.openrouter_api_key = api_key
        elif provider == "azure":
            config.azure_model = model
            if api_key:
                config.azure_api_key = api_key
            if endpoint:
                config.azure_endpoint = endpoint
        elif provider == "google":
            config.google_model = model
            if api_key:
                config.google_api_key = api_key
        elif provider == "local":
            config.local_llm_model = model
            if endpoint:
                config.local_llm_endpoint = endpoint
        
        return {
            "status": "success",
            "message": f"LLM provider configured: {provider}",
            "provider": provider,
            "model": model,
            "endpoint": endpoint if endpoint else "default"
        }
        
    except Exception as e:
        return {
            "error": f"Failed to configure LLM provider: {str(e)}"
        }

@mcp.tool()
async def list_llm_providers() -> Dict[str, Any]:
    """List all available LLM providers and their current configuration"""
    
    providers_info = {
        "available_providers": {
            "groq": {
                "description": "GROQ API for fast inference",
                "models": ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
                "requires_api_key": True,
                "current_model": config.groq_model,
                "configured": bool(config.groq_api_key)
            },
            "openai": {
                "description": "OpenAI GPT models",
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                "requires_api_key": True,
                "current_model": config.openai_model,
                "configured": bool(config.openai_api_key)
            },
            "anthropic": {
                "description": "Anthropic Claude models",
                "models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
                "requires_api_key": True,
                "current_model": config.anthropic_model,
                "configured": bool(config.anthropic_api_key)
            },
            "openrouter": {
                "description": "OpenRouter - Access to multiple models",
                "models": ["deepseek/deepseek-chat", "anthropic/claude-3.5-sonnet", "google/gemini-2.0-flash-exp", "meta-llama/llama-3.1-70b-instruct"],
                "requires_api_key": True,
                "current_model": config.openrouter_model,
                "configured": bool(config.openrouter_api_key)
            },
            "azure": {
                "description": "Azure OpenAI Service",
                "models": ["gpt-4o", "gpt-4-turbo", "gpt-35-turbo"],
                "requires_api_key": True,
                "requires_endpoint": True,
                "current_model": config.azure_model,
                "configured": bool(config.azure_api_key and config.azure_endpoint)
            },
            "google": {
                "description": "Google Gemini models",
                "models": ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
                "requires_api_key": True,
                "current_model": config.google_model,
                "configured": bool(config.google_api_key)
            },
            "local": {
                "description": "Local LLM (Ollama, LM Studio, etc.)",
                "models": ["llama3.2", "codellama", "mistral", "custom"],
                "requires_api_key": False,
                "requires_endpoint": True,
                "current_model": config.local_llm_model,
                "current_endpoint": config.local_llm_endpoint,
                "configured": True
            },
            "vscode_lm": {
                "description": "VS Code Language Model API (experimental)",
                "models": ["depends on VS Code extensions"],
                "requires_api_key": False,
                "current_model": "N/A",
                "configured": False,
                "note": "Requires VS Code extension context"
            }
        },
        "current_provider": config.default_llm_provider.value,
        "integration_notes": {
            "cursor": "Use custom API keys or tunnel local endpoints",
            "cline": "Supports VS Code LM API and multiple providers",
            "windsurf": "Native MCP integration - add server to mcp_config.json",
            "roo_code": "Supports OpenRouter, OpenAI, Anthropic, and local models"
        }
    }
    
    return providers_info

@mcp.tool()
async def query_with_rag(
    query: str = Field(description="The simple prompt to enhance using Self-RAG and prompt engineering techniques"),
    repo_id: Optional[str] = Field(description="Repository ID to search in (optional)", default=None),
    use_local_llm: bool = Field(description="Whether to use local LLM instead of GROQ", default=False)
) -> Dict[str, Any]:
    """Enhance a simple prompt using Self-RAG pipeline with code context and advanced prompt engineering techniques"""
    
    await ensure_initialized()
    try:
        # Normalize repo_id parameter (handle FieldInfo objects from FastMCP)
        if repo_id is not None and not isinstance(repo_id, str):
            repo_id = None
        # Auto-scan and upload workspace first
        print("🔍 Auto-scanning workspace for code context...")
        workspace_id = await scan_and_upload_workspace()
        
        # Use workspace repo if no specific repo provided
        if repo_id is None and workspace_id:
            repo_id = workspace_id
            print(f"📁 Using auto-uploaded workspace: {workspace_id}")
        
        # Create LLM configuration
        llm_config = LLMConfig(
            provider=LLMProvider.LOCAL if use_local_llm else LLMProvider.GROQ,
            model=config.local_llm_model if use_local_llm else config.groq_model,
            api_key=config.groq_api_key,
            endpoint=config.local_llm_endpoint
        )
        
        # Validate repository if specified (but workspace repos are automatically valid)
        if repo_id and not str(repo_id).startswith("workspace_") and repo_id not in active_repositories:
            return {
                "error": f"Repository {repo_id} not found. Use list_repositories to see available repos.",
                "available_repos": list(active_repositories.keys())
            }
        
        # Execute the complete Self-RAG pipeline
        results = await self_rag_pipeline(query, repo_id, llm_config)
        
        # Format response for MCP - return the enhanced prompt
        response = {
            "enhanced_prompt": results.get("enhanced_prompt", results.get("final_answer", "No enhanced prompt generated")),
            "original_query": query,
            "pipeline_steps": results.get("steps", []),
            "technique_selection": results.get("technique_selection", {}),
            "documents_used": results.get("documents_used", 0),
            "web_results_used": results.get("web_results_used", 0),
            "processing_time": results.get("processing_time", 0),
            "repo_searched": repo_id or "All repositories"
        }
        
        if "error" in results:
            response["error"] = results["error"]
        
        return response
        
    except Exception as e:
        return {
            "error": f"Self-RAG query failed: {str(e)}",
            "query": query,
            "repo_id": repo_id
        }

@mcp.tool()
async def test_llm_connection(
    use_local_llm: bool = Field(description="Test local LLM instead of GROQ", default=False),
    test_query: str = Field(description="Test query to send", default="Hello, how are you?")
) -> Dict[str, Any]:
    """Test LLM connection and get a simple response"""
    
    try:
        llm_config = LLMConfig(
            provider=LLMProvider.LOCAL if use_local_llm else LLMProvider.GROQ,
            model=config.local_llm_model if use_local_llm else config.groq_model,
            api_key=config.groq_api_key,
            endpoint=config.local_llm_endpoint
        )
        
        response = await call_llm(test_query, llm_config)
        
        return {
            "success": True,
            "provider": llm_config.provider.value,
            "model": llm_config.model,
            "response": response[:200] + "..." if len(response) > 200 else response
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "provider": "local" if use_local_llm else "groq"
        }

@mcp.tool()
async def search_repository_documents(
    query: str = Field(description="Search query for finding relevant documents"),
    repo_id: Optional[str] = Field(description="Repository ID to search in (optional)", default=None),
    limit: int = Field(description="Maximum number of documents to return", default=5)
) -> Dict[str, Any]:
    """Search for documents in repositories using vector similarity"""
    
    try:
        # Normalize repo_id parameter (handle FieldInfo objects from FastMCP)
        if repo_id is not None and not isinstance(repo_id, str):
            repo_id = None
        # Auto-scan and upload workspace first
        workspace_id = await scan_and_upload_workspace()
        
        # Use workspace repo if no specific repo provided
        if repo_id is None and workspace_id:
            repo_id = workspace_id
        
        # Validate repository if specified (but workspace repos are automatically valid)
        if repo_id and not str(repo_id).startswith("workspace_") and repo_id not in active_repositories:
            return {
                "error": f"Repository {repo_id} not found",
                "available_repos": list(active_repositories.keys())
            }
        
        # Search documents
        documents = await search_documents(query, repo_id, limit)
        
        # Format results
        formatted_docs = []
        for doc in documents:
            formatted_docs.append({
                "file_path": doc['file_path'],
                "chunk_index": doc['chunk_index'],
                "similarity_score": round(doc['score'], 4),
                "content_preview": doc['text'][:300] + "..." if len(doc['text']) > 300 else doc['text'],
                "full_content": doc['text']
            })
        
        return {
            "query": query,
            "repo_searched": repo_id or "All repositories",
            "documents_found": len(formatted_docs),
            "documents": formatted_docs
        }
        
    except Exception as e:
        return {
            "error": f"Document search failed: {str(e)}",
            "query": query,
            "repo_id": repo_id
        }

@mcp.tool()
async def enhance_prompt(
    simple_prompt: str = Field(description="The simple prompt to enhance with code context and techniques"),
    repo_id: Optional[str] = Field(description="Repository ID to search for relevant code (optional)", default=None)
) -> Dict[str, Any]:
    """Transform a simple prompt into an enhanced prompt with relevant code context and optimal prompt engineering techniques"""
    
    try:
        # Normalize repo_id parameter (handle FieldInfo objects from FastMCP)
        if repo_id is not None and not isinstance(repo_id, str):
            repo_id = None
        # Auto-scan and upload workspace first
        print("🔍 Auto-scanning workspace for code context...")
        workspace_id = await scan_and_upload_workspace()
        
        # Use workspace repo if no specific repo provided
        if repo_id is None and workspace_id:
            repo_id = workspace_id
            print(f"📁 Using auto-uploaded workspace: {workspace_id}")
        
        # Execute the Self-RAG pipeline to get enhanced prompt
        results = await self_rag_pipeline(simple_prompt, repo_id, None)
        
        enhanced_prompt = results.get("enhanced_prompt", results.get("final_answer", ""))
        
        return {
            "enhanced_prompt": enhanced_prompt,
            "original_prompt": simple_prompt,
            "techniques_applied": results.get("technique_selection", {}).get("selected_techniques", []),
            "code_snippets_included": results.get("documents_used", 0),
            "web_context_added": results.get("web_results_used", 0),
            "processing_time": results.get("processing_time", 0)
        }
        
    except Exception as e:
        return {
            "error": f"Prompt enhancement failed: {str(e)}",
            "original_prompt": simple_prompt
        }

@mcp.tool()
async def test_web_search(
    query: str = Field(description="Query to search for on the web"),
    max_results: int = Field(description="Maximum number of results to return", default=3)
) -> Dict[str, Any]:
    """Test web search functionality using Tavily API"""
    
    try:
        if not config.tavily_api_key:
            return {
                "error": "Tavily API key not configured. Set TAVILY_API_KEY environment variable."
            }
        
        web_results = await web_search_tavily(query, max_results)
        
        if not web_results:
            return {
                "query": query,
                "results_found": 0,
                "message": "No web search results found or web search not available"
            }
        
        # Format results
        formatted_results = []
        for result in web_results:
            formatted_results.append({
                "title": result['title'],
                "url": result['url'],
                "content_preview": result['content'][:300] + "..." if len(result['content']) > 300 else result['content'],
                "score": result['score']
            })
        
        return {
            "query": query,
            "results_found": len(formatted_results),
            "web_results": formatted_results
        }
        
    except Exception as e:
        return {
            "error": f"Web search test failed: {str(e)}",
            "query": query
        }

@mcp.tool()
async def test_code_retrieval(
    query: str = Field(description="Query to test code snippet retrieval"),
    repo_id: Optional[str] = Field(description="Repository ID to search in (optional)", default=None),
    limit: int = Field(description="Maximum number of code snippets to return", default=5)
) -> Dict[str, Any]:
    """Test code snippet retrieval functionality with detailed debug info"""
    
    await ensure_initialized()
    
    try:
        # Normalize repo_id parameter (handle FieldInfo objects from FastMCP)
        if repo_id is not None and not isinstance(repo_id, str):
            repo_id = None
        print(f"🧪 Testing code retrieval for query: '{query}'")
        
        # Step 1: Search for documents
        documents = await search_documents(query, repo_id, limit)
        
        # Step 2: Grade relevance of each document
        graded_docs = []
        for doc in documents:
            grade = await grade_document_relevance(doc['text'], query)
            doc_with_grade = {
                **doc,
                'relevance_grade': grade
            }
            graded_docs.append(doc_with_grade)
        
        # Step 3: Filter relevant documents
        relevant_docs = [doc for doc in graded_docs if doc['relevance_grade']['relevant']]
        
        # Prepare detailed results
        results = {
            "success": True,
            "query": query,
            "repo_id": repo_id,
            "search_results": {
                "total_found": len(documents),
                "after_relevance_filtering": len(relevant_docs),
                "collections_searched": list(set(doc.get('collection', 'unknown') for doc in documents))
            },
            "documents": []
        }
        
        # Include detailed document info for debugging
        for doc in graded_docs[:limit]:
            doc_info = {
                "file_path": doc['file_path'],
                "score": doc['score'],
                "chunk_index": doc['chunk_index'],
                "relevant": doc['relevance_grade']['relevant'],
                "relevance_reasoning": doc['relevance_grade']['reasoning'],
                "similarity_score": doc['relevance_grade'].get('similarity_score', 'N/A'),
                "text_preview": doc['text'][:300] + "..." if len(doc['text']) > 300 else doc['text'],
                "text_length": len(doc['text'])
            }
            results["documents"].append(doc_info)
        
        # Summary stats
        if documents:
            avg_score = sum(doc['score'] for doc in documents) / len(documents)
            results["summary"] = {
                "average_search_score": round(avg_score, 3),
                "top_file": documents[0]['file_path'] if documents else None,
                "relevance_pass_rate": f"{len(relevant_docs)}/{len(documents)} ({len(relevant_docs)/len(documents)*100:.1f}%)" if documents else "0/0"
            }
        
        return results
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

# ----------------------
# Server Initialization
# ----------------------

# Initialize server components at module level
_initialized = False

async def ensure_initialized():
    """Ensure server components are initialized"""
    global _initialized
    if not _initialized:
        print("🚀 Starting MCP Self-RAG Prompt Engineering Server...")
        await initialize_components()
        _initialized = True

if __name__ == "__main__":
    # Initialize components before starting server
    import asyncio
    asyncio.run(ensure_initialized())
    mcp.run() 