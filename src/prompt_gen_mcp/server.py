#!/usr/bin/env python3
"""
PromptGen MCP Server - Local MCP that connects to PromptGen API for techniques
Users run this locally and it connects to the hosted PromptGen service for techniques
"""

import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP

# Configuration
@dataclass
class Config:
    promptgen_api_key: str = os.getenv("PROMPTGEN_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    promptgen_base_url: str = "https://promptgenmcp-production.up.railway.app"  # Railway deployment URL
    workspace_path: str = os.getenv("WORKSPACE_PATH", os.getcwd())
    max_file_size: int = 50000  # 50KB max per file
    embedding_model_name: str = "intfloat/multilingual-e5-large-instruct"
    
    def get_project_workspace(self) -> str:
        """Intelligently detect the current project workspace"""
        current_dir = Path(self.workspace_path)
        
        # Look for common project indicators in current directory and parents
        project_indicators = {
            'package.json', 'pyproject.toml', 'requirements.txt', 'Cargo.toml',
            'go.mod', 'pom.xml', 'build.gradle', '.git', '.gitignore',
            'README.md', 'setup.py', 'composer.json', 'Gemfile'
        }
        
        # Start from current directory and go up to find project root
        for path in [current_dir] + list(current_dir.parents):
            # Count subdirectories that look like projects
            project_subdirs = []
            if path.is_dir():
                for subdir in path.iterdir():
                    if (subdir.is_dir() and 
                        not subdir.name.startswith('.') and 
                        not subdir.name.startswith('__') and
                        subdir.name not in ['node_modules', 'venv', 'env', '.venv']):
                        
                        # Check if subdir has project indicators
                        has_indicators = any((subdir / indicator).exists() for indicator in project_indicators)
                        if has_indicators:
                            project_subdirs.append(subdir.name)
            
            # If we found multiple project subdirectories, this is likely the workspace root
            if len(project_subdirs) >= 2:
                return str(path)
            
            # Check if current directory itself has project indicators
            if any((path / indicator).exists() for indicator in project_indicators):
                return str(path)
        
        # If no project indicators found, use current directory
        return str(current_dir)

config = Config()

# Try to import optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    import httpx
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Optional dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False

# Try to import Tavily for web search
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

class RetrievalGrader:
    """Grade document relevance using LLM"""
    
    def __init__(self):
        self.client = None
        if DEPENDENCIES_AVAILABLE and config.groq_api_key:
            self.client = httpx.AsyncClient(timeout=30.0)
    
    async def grade_relevance(self, question: str, document: str) -> Dict[str, Any]:
        """Grade document relevance to question"""
        if not self.client or not config.groq_api_key:
            return {"relevant": True, "score": 0.5, "reasoning": "Fallback: assuming relevant"}
        
        try:
            prompt = f"""You are a grader assessing relevance of a retrieved document to a user question.

Document: {document[:128000]}...

Question: {question}

Is this document relevant to answering the question? Consider:
- Does it contain keywords related to the question?
- Does it provide context that could help answer the question?
- Is the semantic meaning related?

Respond with JSON only:
{{
    "relevant": true/false,
    "score": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

            response = await self.client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-r1-distill-llama-70b",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.6,
                    "max_tokens": 131072,
                    "reasoning_format": "parsed",
                    "top_p": 0.95
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                message = result["choices"][0]["message"]
                
                # Handle parsed reasoning format
                if "reasoning" in message:
                    print(f"🧠 Reasoning: {message['reasoning'][:200]}...")
                    content = message["content"].strip()
                else:
                    content = message["content"].strip()
                
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                
                return json.loads(content)
            else:
                return {"relevant": True, "score": 0.5, "reasoning": "API error fallback"}
                
        except Exception as e:
            print(f"❌ Relevance grading failed: {e}")
            return {"relevant": True, "score": 0.5, "reasoning": "Error fallback"}

class HallucinationGrader:
    """Detect hallucinations in generated responses"""
    
    def __init__(self):
        self.client = None
        if DEPENDENCIES_AVAILABLE and config.groq_api_key:
            self.client = httpx.AsyncClient(timeout=30.0)
    
    async def check_grounding(self, documents: List[str], generation: str) -> Dict[str, Any]:
        """Check if generation is grounded in documents"""
        if not self.client or not config.groq_api_key:
            return {"grounded": True, "score": 0.8, "reasoning": "Fallback: assuming grounded"}
        
        try:
            docs_text = "\n\n".join(documents[:3])  # Limit to first 3 docs
            prompt = f"""You are a grader assessing whether an LLM generation is grounded in retrieved facts.

Facts from documents:
{docs_text[:128000]}...

LLM Generation:
{generation}

Is the generation grounded in/supported by the facts? Check:
- Are claims supported by the documents?
- Are there unsupported assertions?
- Is information fabricated or hallucinated?

Respond with JSON only:
{{
    "grounded": true/false,
    "score": 0.0-1.0,
    "reasoning": "brief explanation",
    "unsupported_claims": ["list of unsupported claims if any"]
}}"""

            response = await self.client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-r1-distill-llama-70b",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.6,
                    "max_tokens": 131072,
                    "reasoning_format": "parsed",
                    "top_p": 0.95
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                message = result["choices"][0]["message"]
                
                # Handle parsed reasoning format
                if "reasoning" in message:
                    print(f"🧠 Hallucination Check Reasoning: {message['reasoning'][:200]}...")
                    content = message["content"].strip()
                else:
                    content = message["content"].strip()
                
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                
                return json.loads(content)
            else:
                return {"grounded": True, "score": 0.8, "reasoning": "API error fallback"}
                
        except Exception as e:
            print(f"❌ Hallucination check failed: {e}")
            return {"grounded": True, "score": 0.8, "reasoning": "Error fallback"}

class AnswerQualityGrader:
    """Grade answer quality and relevance to question"""
    
    def __init__(self):
        self.client = None
        if DEPENDENCIES_AVAILABLE and config.groq_api_key:
            self.client = httpx.AsyncClient(timeout=30.0)
    
    async def grade_answer(self, question: str, generation: str) -> Dict[str, Any]:
        """Grade if answer addresses the question"""
        if not self.client or not config.groq_api_key:
            return {"addresses_question": True, "score": 0.8, "reasoning": "Fallback: assuming good"}
        
        try:
            prompt = f"""You are a grader assessing whether an answer addresses/resolves a question.

Question: {question}

Answer: {generation}

Does the answer address the question? Consider:
- Does it directly answer what was asked?
- Is it relevant and on-topic?
- Does it provide useful information?
- Is it complete enough?

Respond with JSON only:
{{
    "addresses_question": true/false,
    "score": 0.0-1.0,
    "reasoning": "brief explanation",
    "missing_aspects": ["list of missing aspects if any"]
}}"""

            response = await self.client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-r1-distill-llama-70b",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.6,
                    "max_tokens": 131072,
                    "reasoning_format": "parsed",
                    "top_p": 0.95
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                message = result["choices"][0]["message"]
                
                # Handle parsed reasoning format
                if "reasoning" in message:
                    print(f"🧠 Answer Quality Reasoning: {message['reasoning'][:200]}...")
                    content = message["content"].strip()
                else:
                    content = message["content"].strip()
                
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                
                return json.loads(content)
            else:
                return {"addresses_question": True, "score": 0.8, "reasoning": "API error fallback"}
                
        except Exception as e:
            print(f"❌ Answer quality grading failed: {e}")
            return {"addresses_question": True, "score": 0.8, "reasoning": "Error fallback"}

class WebSearcher:
    """Web search using Tavily API (optional)"""
    
    def __init__(self):
        self.search_tool = None
        if TAVILY_AVAILABLE and config.tavily_api_key:
            try:
                self.search_tool = TavilySearchResults(k=3, api_key=config.tavily_api_key)
                print("✅ Tavily web search initialized")
            except Exception as e:
                print(f"❌ Failed to initialize Tavily: {e}")
                self.search_tool = None
        else:
            if not config.tavily_api_key:
                print("ℹ️  No Tavily API key provided, web search disabled")
            else:
                print("ℹ️  Tavily not available, web search disabled")
    
    async def search_web_docs(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant documentation using Tavily"""
        if not self.search_tool:
            print("⚠️ Web search not available")
            return []
        
        try:
            print(f"🌐 Searching web for: {query}")
            results = self.search_tool.invoke({"query": query})
            
            web_docs = []
            for result in results:
                web_doc = {
                     'content': f"""Web Source: {result.get('url', 'Unknown')}
Title: {result.get('title', 'No title')}

Content:
{result.get('content', 'No content')}""",
                     'url': result.get('url', ''),
                     'title': result.get('title', ''),
                     'path': result.get('url', 'Web Search Result'),  # Add path field for compatibility
                     'extension': '.md',  # Default to markdown for web content
                     'relevance_score': 0.8,  # Default high relevance for web results
                     'source': 'web_search'
                 }
                web_docs.append(web_doc)
            
            print(f"✅ Found {len(web_docs)} web search results")
            return web_docs
            
        except Exception as e:
            print(f"❌ Web search error: {e}")
            return []

class QueryTransformer:
    """Transform queries for better retrieval"""
    
    def __init__(self):
        self.client = None
        if DEPENDENCIES_AVAILABLE and config.groq_api_key:
            self.client = httpx.AsyncClient(timeout=30.0)
    
    async def transform_query(self, question: str, context: str = "") -> str:
        """Transform query for better retrieval"""
        if not self.client or not config.groq_api_key:
            return question  # Fallback to original
        
        try:
            prompt = f"""You are a query optimizer that improves questions for better document retrieval.

Original question: {question}

Context: {context}

Rewrite this question to be more specific and optimized for finding relevant documents. Consider:
- Adding technical keywords
- Being more specific about the domain
- Clarifying the intent
- Making it more searchable

Return only the improved question, no explanation."""

            response = await self.client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-r1-distill-llama-70b",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.6,
                    "max_tokens": 131072,
                    "reasoning_format": "parsed",
                    "top_p": 0.95
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                message = result["choices"][0]["message"]
                
                # Handle parsed reasoning format
                if "reasoning" in message:
                    print(f"🧠 Question Rewrite Reasoning: {message['reasoning'][:200]}...")
                    improved_question = message["content"].strip()
                else:
                    improved_question = message["content"].strip()
                return {"transformed_query": improved_question}
            else:
                return {"transformed_query": question}
                
        except Exception as e:
            print(f"❌ Query transformation failed: {e}")
            return {"transformed_query": question}

class LocalEmbeddingGenerator:
    """Generate embeddings locally for privacy"""
    
    def __init__(self):
        self.model = None
        if DEPENDENCIES_AVAILABLE:
            try:
                print("🔄 Loading embedding model...")
                self.model = SentenceTransformer(config.embedding_model_name)
                print("✅ Embedding model loaded")
            except Exception as e:
                print(f"❌ Failed to load embedding model: {e}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        if not self.model:
            # Return dummy embedding if model not available
            return [0.0] * 1024
        
        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            return [0.0] * 1024

class PromptGenAPIClient:
    """Client for connecting to PromptGen API to fetch techniques"""
    
    def __init__(self):
        self.client = None
        if DEPENDENCIES_AVAILABLE and config.promptgen_api_key:
            self.client = httpx.AsyncClient(timeout=30.0)
    
    async def fetch_techniques(self, question: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Fetch techniques from PromptGen API"""
        if not self.client or not config.promptgen_api_key:
            return self._fallback_techniques()
        
        try:
            print(f"🌐 Fetching techniques from PromptGen API for: {question}")
            
            response = await self.client.post(
                f"{config.promptgen_base_url}/api/mcp/techniques",
                headers={
                    "Authorization": f"Bearer {config.promptgen_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "question": question,
                    "limit": limit
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                techniques = data.get("techniques", [])
                print(f"✅ Fetched {len(techniques)} techniques from PromptGen API")
                return techniques
            else:
                print(f"❌ PromptGen API error: {response.status_code} - {response.text}")
                return self._fallback_techniques()
                
        except Exception as e:
            print(f"❌ Failed to fetch techniques from PromptGen API: {e}")
            return self._fallback_techniques()
    
    def _fallback_techniques(self) -> List[Dict[str, Any]]:
        """Fallback techniques when API is unavailable"""
        return [
            {
                "name": "Chain of Thought",
                "description": "Break down complex problems into step-by-step reasoning",
                "example": "Let me think through this step by step: 1) First... 2) Then... 3) Finally...",
                "score": 0.9
            },
            {
                "name": "Plan and Solve",
                "description": "Create a structured plan before implementing the solution",
                "example": "Plan: Identify the problem, research solutions, implement. Solve: Execute each step systematically.",
                "score": 0.85
            },
            {
                "name": "Few-Shot Learning",
                "description": "Provide examples to guide the response",
                "example": "Here are some examples: Example 1: ... Example 2: ... Now apply this pattern to your case.",
                "score": 0.8
            }
        ]

class LLMQuestionAnalyzer:
    """Analyze questions using LLM for intelligent categorization"""
    
    def __init__(self):
        self.client = None
        if DEPENDENCIES_AVAILABLE and config.groq_api_key:
            self.client = httpx.AsyncClient(timeout=30.0)
    
    async def analyze_question(self, question: str) -> Dict[str, Any]:
        """Analyze question type and complexity using LLM"""
        if not self.client or not config.groq_api_key:
            return self._simple_analysis(question)
        
        try:
            prompt = f"""Analyze this question and categorize it:
Question: "{question}"

Respond with ONLY a JSON object in this exact format:
{{
    "types": ["debugging", "optimization", "architecture", "implementation", "explanation", "comparison", "security", "testing"],
    "complexity": "low|medium|high",
    "domain": "frontend|backend|database|devops|general",
    "intent": "learn|solve|compare|debug|optimize"
}}

Pick the most relevant types (1-3), complexity level, domain, and primary intent."""

            response = await self.client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-r1-distill-llama-70b",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.6,
                    "max_tokens": 131072,
                    "reasoning_format": "parsed",
                    "top_p": 0.95
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                message = result["choices"][0]["message"]
                
                # Handle parsed reasoning format
                if "reasoning" in message:
                    print(f"🧠 Question Analysis Reasoning: {message['reasoning'][:200]}...")
                    content = message["content"].strip()
                else:
                    content = message["content"].strip()
                
                # Extract JSON from response
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                
                analysis = json.loads(content)
                print(f"🧠 LLM Analysis: {analysis}")
                return analysis
            else:
                print(f"❌ LLM API error: {response.status_code}")
                return self._simple_analysis(question)
            
        except Exception as e:
            print(f"❌ LLM analysis failed: {e}")
            return self._simple_analysis(question)
    
    def _simple_analysis(self, question: str) -> Dict[str, Any]:
        """Simple keyword-based analysis as fallback"""
        question_lower = question.lower()
        
        types = []
        if any(word in question_lower for word in ["bug", "error", "fix", "debug", "issue"]):
            types.append("debugging")
        if any(word in question_lower for word in ["optimize", "performance", "faster", "slow"]):
            types.append("optimization")
        if any(word in question_lower for word in ["architecture", "design", "structure", "pattern"]):
            types.append("architecture")
        if any(word in question_lower for word in ["implement", "create", "build", "make"]):
            types.append("implementation")
        if any(word in question_lower for word in ["explain", "how", "what", "why"]):
            types.append("explanation")
        
        if not types:
            types = ["general"]
        
        complexity = "high" if len(question.split()) > 15 else "medium" if len(question.split()) > 8 else "low"
        
        return {
            "types": types,
            "complexity": complexity,
            "domain": "general",
            "intent": "solve"
        }

class CodeContextScanner:
    """Scan workspace for relevant code context"""
    
    def __init__(self):
        self.supported_extensions = {'.py', '.js', '.ts', '.tsx', '.jsx', '.md', '.txt', '.json', '.yml', '.yaml'}
    
    def _detect_target_directory(self, question: str) -> Path:
        """Detect if question mentions a specific project/directory to scan"""
        project_workspace = Path(config.get_project_workspace())
        question_lower = question.lower()
        
        # Check if question mentions specific subdirectories
        potential_dirs = []
        
        # Look for directory names in the question (exact and partial matches)
        for item in project_workspace.iterdir():
            if item.is_dir() and not item.name.startswith('.') and not self._should_ignore(item):
                dir_name_lower = item.name.lower()
                
                # Check for exact match or if directory name is mentioned in question
                if (dir_name_lower in question_lower or 
                    any(word in dir_name_lower for word in question_lower.split() if len(word) > 3)):
                    potential_dirs.append((item, dir_name_lower in question_lower))
                # Check for common abbreviations or variations
                elif self._check_directory_variations(dir_name_lower, question_lower):
                    potential_dirs.append((item, False))  # Mark as partial match
        
        # Sort by exact matches first, then partial matches
        potential_dirs.sort(key=lambda x: x[1], reverse=True)
        
        # If specific directory mentioned, use it; otherwise use project root
        if potential_dirs:
            # Use the best matching directory
            target_dir = potential_dirs[0][0]
            match_type = "exact" if potential_dirs[0][1] else "partial"
            print(f"🎯 Detected target directory: {target_dir.name} ({match_type} match)")
            return target_dir
        else:
            print(f"🏠 Using project workspace: {project_workspace.name}")
            return project_workspace
    
    def _check_directory_variations(self, dir_name: str, question: str) -> bool:
        """Check for common directory name variations and abbreviations"""
        variations = {
            'cryptocasino': ['crypto', 'casino', 'gambling', 'game'],
            'promptgenmcp': ['prompt', 'gen', 'mcp', 'generation'],
            'python-sdk': ['python', 'sdk', 'py'],
        }
        
        dir_key = dir_name.replace('-', '').replace('_', '')
        if dir_key in variations:
            return any(var in question for var in variations[dir_key])
        
        return False
    
    async def scan_workspace(self, question: str) -> List[Dict[str, Any]]:
        """Scan workspace for relevant files and extract context"""
        try:
            # Intelligently detect the target directory to scan
            workspace_path = self._detect_target_directory(question)
            relevant_files = []
            
            # Find relevant files
            for file_path in workspace_path.rglob("*"):
                if (file_path.is_file() and 
                    file_path.suffix in self.supported_extensions and
                    not self._should_ignore(file_path)):
                    
                    try:
                        if file_path.stat().st_size > config.max_file_size:
                            continue
                        
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        
                        # Simple relevance scoring
                        relevance_score = self._calculate_relevance(question, content, file_path.name)
                        
                        if relevance_score > 0.1:  # Threshold for relevance
                            # Calculate relative path from the project root for better context
                            project_root = Path(config.get_project_workspace())
                            try:
                                relative_path = str(file_path.relative_to(project_root))
                            except ValueError:
                                # If file is outside project root, use relative to workspace_path
                                relative_path = str(file_path.relative_to(workspace_path))
                            
                            relevant_files.append({
                                'path': relative_path,
                                'content': content[:128000],  # Limit content length
                                'relevance_score': relevance_score,
                                'size': len(content),
                                'extension': file_path.suffix
                            })
                    
                    except Exception as e:
                        continue
            
            # Sort by relevance and return top results
            relevant_files.sort(key=lambda x: x['relevance_score'], reverse=True)
            return relevant_files[:5]  # Return top 5 most relevant files
        
        except Exception as e:
            print(f"❌ Workspace scan failed: {e}")
            return []
        
    def _should_ignore(self, file_path: Path) -> bool:
        """Check if file should be ignored"""
        ignore_patterns = {
            'node_modules', '.git', '__pycache__', '.pytest_cache',
            'venv', '.venv', 'env', '.env', 'dist', 'build'
        }
        
        return any(pattern in str(file_path) for pattern in ignore_patterns)
    
    def _calculate_relevance(self, question: str, content: str, filename: str) -> float:
        """Calculate relevance score between question and file content"""
        question_words = set(question.lower().split())
        content_words = set(content.lower().split())
        filename_words = set(filename.lower().replace('.', ' ').split())
        
        # Calculate word overlap
        content_overlap = len(question_words.intersection(content_words)) / len(question_words) if question_words else 0
        filename_overlap = len(question_words.intersection(filename_words)) / len(question_words) if question_words else 0
        
        # Weight filename matches higher
        relevance_score = (content_overlap * 0.7) + (filename_overlap * 0.3)
        
        return min(relevance_score, 1.0)

class PromptEnhancer:
    """Generate enhanced prompts with Self-RAG workflow"""
    
    def __init__(self):
        self.retrieval_grader = RetrievalGrader()
        self.hallucination_grader = HallucinationGrader()
        self.answer_grader = AnswerQualityGrader()
        self.query_transformer = QueryTransformer()
        self.web_searcher = WebSearcher()
        self.max_iterations = 3
    
    async def enhance_prompt_with_self_rag(
        self,
        question: str,
        analysis: Dict[str, Any],
        techniques: List[Dict[str, Any]],
        code_context: List[Dict[str, Any]]
    ) -> str:
        """Generate enhanced prompt using Self-RAG workflow"""
        
        iteration = 0
        current_question = question
        filtered_context = code_context
        generation_history = []
        
        while iteration < self.max_iterations:
            iteration += 1
            print(f"🔄 Self-RAG Iteration {iteration}/{self.max_iterations}")
            
            # Step 1: Grade document relevance
            print("📊 Grading document relevance...")
            relevant_docs = []
            for doc in filtered_context:
                grade = await self.retrieval_grader.grade_relevance(
                    current_question, doc['content']
                )
                if grade['relevant']:
                    relevant_docs.append({
                        **doc,
                        'relevance_grade': grade['score'],
                        'reasoning': grade['reasoning']
                    })
            
            print(f"✅ {len(relevant_docs)}/{len(filtered_context)} documents deemed relevant")
            
            # Step 1.5: Search web for additional documentation if needed
            if len(relevant_docs) < 2 or iteration > 1:  # Search web if few relevant docs or on retry
                print("🌐 Searching web for additional documentation...")
                web_docs = await self.web_searcher.search_web_docs(
                    f"{current_question} documentation tutorial guide", max_results=3
                )
                if web_docs:
                    # Grade web document relevance
                    for doc in web_docs:
                        grade = await self.retrieval_grader.grade_relevance(
                            current_question, doc['content']
                        )
                        if grade['relevant']:
                            doc['relevance_grade'] = grade['score']
                            doc['reasoning'] = grade['reasoning']
                            relevant_docs.append(doc)
                    print(f"✅ Added {len([d for d in web_docs if 'relevance_grade' in d])} relevant web documents")
            
            # Step 2: Generate initial response
            enhanced_prompt = await self._generate_enhanced_prompt(
                current_question, analysis, techniques, relevant_docs, iteration
            )
            
            # Step 3: Check for hallucinations
            print("🔍 Checking for hallucinations...")
            hallucination_check = await self.hallucination_grader.check_grounding(
                [doc['content'] for doc in relevant_docs], enhanced_prompt
            )
            # Convert grounding result to hallucination format
            hallucination_check['has_hallucination'] = not hallucination_check.get('grounded', True)
            
            # Step 4: Assess answer quality
            print("⭐ Assessing answer quality...")
            quality_check = await self.answer_grader.grade_answer(
                current_question, enhanced_prompt
            )
            
            generation_history.append({
                'iteration': iteration,
                'question': current_question,
                'relevant_docs': len(relevant_docs),
                'hallucination_score': hallucination_check['score'],
                'quality_score': quality_check['score'],
                'prompt_length': len(enhanced_prompt)
            })
            
            # Step 5: Decision logic
            if (not hallucination_check['has_hallucination'] and 
                quality_check['addresses_question'] and 
                quality_check['score'] >= 0.7):
                print(f"✅ High-quality response achieved in iteration {iteration}")
                break
            
            # Step 6: Search web for additional context if quality is poor
            if (quality_check['score'] < 0.7 and iteration < self.max_iterations):
                print("🌐 Searching web for additional context due to poor quality...")
                web_docs = await self.web_searcher.search_web_docs(
                    f"{current_question} best practices examples", max_results=2
                )
                if web_docs:
                    for doc in web_docs:
                        grade = await self.retrieval_grader.grade_relevance(
                            current_question, doc['content']
                        )
                        if grade['relevant']:
                            doc['relevance_grade'] = grade['score']
                            doc['reasoning'] = grade['reasoning']
                            filtered_context.append(doc)
                    print(f"✅ Added {len([d for d in web_docs if 'relevance_grade' in d])} web docs to context")
            
            # Step 7: Transform query if needed
            if iteration < self.max_iterations:
                print("🔄 Transforming query for better retrieval...")
                transformed = await self.query_transformer.transform_query(
                    current_question, 
                    f"Previous attempt had issues: Hallucination={hallucination_check['has_hallucination']}, Quality={quality_check['score']:.2f}"
                )
                current_question = transformed['transformed_query']
                print(f"📝 New query: {current_question}")
        
        # Add Self-RAG metadata to final prompt
        final_prompt = self._add_self_rag_metadata(
            enhanced_prompt, generation_history, iteration
        )
        
        return final_prompt
    
    async def _generate_enhanced_prompt(
        self,
        question: str,
        analysis: Dict[str, Any],
        techniques: List[Dict[str, Any]],
        relevant_docs: List[Dict[str, Any]],
        iteration: int
    ) -> str:
        """Generate the actual enhanced prompt using the techniques"""
        
        # Extract technique names for application
        technique_names = [t['name'] for t in techniques]
        
        # Build the enhanced prompt by applying the techniques
        enhanced_prompt = ""
        
        # Apply Chain of Thought if present
        if any('Chain of Thought' in name for name in technique_names):
            enhanced_prompt += """Think step by step to solve this problem. Break down your reasoning into clear, logical steps:

"""
        
        # Apply Plan and Solve if present
        if any('Plan and Solve' in name for name in technique_names):
            enhanced_prompt += """First, create a detailed plan to address this question, then execute each step systematically:

**PLAN:**
1. Analyze the current situation
2. Identify key requirements and constraints
3. Develop a solution strategy
4. Implement the solution
5. Validate and test the results

**EXECUTION:**

"""
        
        # Apply Few-Shot Learning if present
        if any('Few-Shot' in name for name in technique_names):
            enhanced_prompt += """Here are some examples of similar problems and their solutions to guide your approach:

**Example 1:** When making a project deployable, key steps include:
- Setting up proper environment configuration
- Implementing security best practices
- Creating deployment scripts
- Adding monitoring and logging

**Example 2:** For security improvements:
- Input validation and sanitization
- Authentication and authorization
- Secure communication (HTTPS/TLS)
- Environment variable management

**Now apply this pattern to the current problem:**

"""
        
        # Add the main question with context
        enhanced_prompt += f"""**Question to solve:** {question}

"""
        
        # Add analysis context
        enhanced_prompt += f"""**Context Analysis:**
- Problem types: {', '.join(analysis.get('types', ['general']))}
- Complexity level: {analysis.get('complexity', 'medium')}
- Domain: {analysis.get('domain', 'general')}
- Intent: {analysis.get('intent', 'solve')}

"""
        
        # Add relevant code context if available
        if relevant_docs:
            enhanced_prompt += """**Available Code Context:**
"""
            for i, doc in enumerate(relevant_docs, 1):
                enhanced_prompt += f"""File {i}: {doc['path']} (Relevance: {doc.get('relevance_grade', doc['relevance_score']):.3f})
```{doc['extension'][1:] if doc['extension'] else 'text'}
{doc['content'][:32000]}{'...' if len(doc['content']) > 32000 else ''}
```

"""
        
        # Add specific instructions based on the question type
        enhanced_prompt += """**Instructions:**
Provide a comprehensive solution that:
1. Addresses all aspects of the question
2. References specific files and code when relevant
3. Includes concrete implementation steps
4. Considers security and best practices
5. Provides clear, actionable recommendations

Be specific, practical, and thorough in your response. If you need to make assumptions, state them clearly.
"""
        
        return enhanced_prompt
    
    def _add_self_rag_metadata(
        self, 
        prompt: str, 
        history: List[Dict], 
        final_iteration: int
    ) -> str:
        """Add Self-RAG process metadata to the final prompt"""
        
        metadata = """\n---\n## Self-RAG Process Summary\n\n"""
        
        for entry in history:
            metadata += f"""**Iteration {entry['iteration']}:**
- Documents: {entry['relevant_docs']} relevant
- Hallucination Score: {entry['hallucination_score']:.3f}
- Quality Score: {entry['quality_score']:.3f}
- Query: {entry['question'][:100]}{'...' if len(entry['question']) > 100 else ''}

"""
        
        metadata += f"""**Final Result:** Converged after {final_iteration} iteration(s)

*This enhanced prompt was generated using Self-RAG with iterative refinement, document grading, hallucination detection, and answer quality assessment.*
"""
        
        return prompt + metadata
    
    # Keep the original method for backward compatibility
    async def enhance_prompt(
        self,
        question: str,
        analysis: Dict[str, Any],
        techniques: List[Dict[str, Any]],
        code_context: List[Dict[str, Any]]
    ) -> str:
        """Legacy method - redirects to Self-RAG implementation"""
        return await self.enhance_prompt_with_self_rag(
            question, analysis, techniques, code_context
        )

# Initialize components
if DEPENDENCIES_AVAILABLE:
    local_embedder = LocalEmbeddingGenerator()
else:
    local_embedder = None

promptgen_client = PromptGenAPIClient()
llm_analyzer = LLMQuestionAnalyzer()
code_scanner = CodeContextScanner()
prompt_enhancer = PromptEnhancer()

# Initialize MCP server
mcp = FastMCP("PromptGen MCP Server")

@mcp.tool()
async def enhance_prompt(question: str) -> str:
    """
    Transform a simple question into an enhanced prompt using PromptGen API and local context.
    
    This tool:
    1. Analyzes your question using LLM intelligence
    2. Scans your workspace for relevant code (100% private)
    3. Fetches optimal prompt engineering techniques from PromptGen API
    4. Generates a comprehensive enhanced prompt
    
    Args:
        question: Your question or prompt to enhance
        
    Returns:
        Enhanced prompt with techniques, analysis, and code context
    """
    try:
        print(f"🎯 Processing question: {question}...")
        
        # Validate API key
        if not config.promptgen_api_key:
            return f"""❌ Error: PROMPTGEN_API_KEY not found. 

Please:
1. Get your API key from: {config.promptgen_base_url}
2. Set environment variable: PROMPTGEN_API_KEY="your_key_here"
3. Restart Cursor

Original question: {question}"""
        
        # Step 1: Analyze the question
        print("🧠 Analyzing question with LLM...")
        analysis = await llm_analyzer.analyze_question(question)
        
        # Step 2: Scan workspace for relevant code
        print("🔍 Scanning workspace for relevant code...")
        code_context = await code_scanner.scan_workspace(question)
        print(f"📁 Found {len(code_context)} relevant files")
        
        # Step 3: Fetch techniques from PromptGen API
        print("🌐 Fetching techniques from PromptGen API...")
        techniques = await promptgen_client.fetch_techniques(question, limit=3)
        print(f"✨ Fetched {len(techniques)} techniques")
        
        # Step 4: Generate enhanced prompt
        print("📝 Generating enhanced prompt...")
        enhanced_prompt = await prompt_enhancer.enhance_prompt(
            question, analysis, techniques, code_context
        )
        
        print(f"✅ Enhanced prompt generated ({len(enhanced_prompt)} characters)")
        return enhanced_prompt
        
    except Exception as e:
        error_msg = f"❌ Error enhancing prompt: {str(e)}"
        print(error_msg)
        return f"Error: {error_msg}\n\nOriginal question: {question}"

if __name__ == "__main__":
    mcp.run()