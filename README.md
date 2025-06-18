# PromptGen MCP Server

A **Model Context Protocol (MCP)** server that enhances your prompts with 47+ advanced prompt engineering techniques, automatically selecting the best techniques based on your question type and providing relevant code context from your workspace.

## What is MCP?

**Model Context Protocol (MCP)** is an open standard that allows AI assistants like Claude, Cursor, and others to connect with external tools and data sources. Think of it as a way to give your AI assistant superpowers by connecting it to specialized services.

This MCP server acts as a "prompt enhancement engine" that:
- 🧠 **Analyzes your questions** using vector similarity search
- 🔍 **Scans your workspace** to find relevant code context (100% private)
- 🎯 **Selects optimal techniques** from 47+ vectorized prompt engineering methods
- ✨ **Generates enhanced prompts** with similarity scores and examples

## Features

### 🚀 **Vectorized Technique Selection**
Uses semantic vector search to choose from 47+ prompt engineering techniques including:
- **Chain of Thought** - Step-by-step reasoning
- **Few-Shot Learning** - Learning from examples  
- **Plan and Solve** - Structured problem solving
- **Self-Ask** - Breaking down complex questions
- **Tree of Thought** - Exploring multiple solution paths
- **And 42 more advanced techniques stored in Qdrant Cloud...**

### 🔍 **Smart Code Context**
- Scans your entire workspace for relevant files
- Extracts pertinent code snippets based on your question
- Includes file paths and line numbers for precise context
- Maintains complete privacy - all code analysis happens locally

### 🎯 **Question-Aware Enhancement**
Analyzes your questions to determine:
- **Question Type**: debugging, optimization, architecture, implementation, etc.
- **Complexity Level**: simple, moderate, high
- **Best Techniques**: automatically selects 2-4 optimal techniques
- **Context Needs**: finds relevant code files and functions

## Quick Setup

### 1. Get API Keys
```bash
# Required API keys:
PROMPTGEN_API_KEY=your_promptgen_key     # Get from promptgen.dev (required)
GROQ_API_KEY=your_groq_key_here          # Free: 6,000 requests/day (optional)
TAVILY_API_KEY=your_tavily_key_here      # Free: 1,000 searches/month (optional)
```

**Important**: The `PROMPTGEN_API_KEY` is **required** to access the vectorized techniques hosted on Qdrant Cloud. Get your key from **promptgen.dev**.

### 2. Install MCP Server
Add this configuration to your `~/.cursor/mcp_servers.json`:

```json
{
  "mcpServers": {
    "prompt-gen": {
      "command": "python",
      "args": ["/path/to/prompt-gen-mcp/src/prompt_gen_mcp/server.py"],
      "env": {
        "GROQ_API_KEY": "your_groq_key_here",
        "TAVILY_API_KEY": "your_tavily_key_here", 
        "PROMPTGEN_API_KEY": "your_promptgen_key_here"
      }
    }
  }
}
```

### 3. Restart Cursor
Completely restart Cursor IDE to load the MCP server.

### 4. Start Enhancing!
Use the **Command Palette** (Cmd/Ctrl+Shift+P) and search for "MCP" to access the `enhance_prompt` tool.

## How It Works

### Input: Simple Question
```
"How do I optimize this React component?"
```

### Output: Enhanced Prompt
```
# Enhanced Prompt with Vectorized Techniques

## Original Request
How do I optimize this React component?

## Selected Techniques (from Qdrant Cloud)

### 1. Plan and Solve (Similarity: 0.847)
**Description**: Break down complex problems into structured planning and systematic solving phases

**Example**: Plan: Identify optimization targets. Solve: Apply specific React patterns.

### 2. Performance Analysis (Similarity: 0.823)
**Description**: Systematic approach to identifying and resolving performance bottlenecks

**Example**: Measure → Analyze → Optimize → Verify performance improvements.

### 3. Code Review Checklist (Similarity: 0.791)
**Description**: Structured approach to reviewing code for common optimization opportunities

**Example**: Check rendering, memory usage, bundle size, and data fetching patterns.

## Local Code Context
## UserProfile.tsx
```tsx
const UserProfile = ({ userId }) => {
  const [user, setUser] = useState(null);
  // ... component code
};
```

## Enhanced Analysis Request
Using the techniques above and the provided code context, please provide a comprehensive response to: "How do I optimize this React component?"

Apply the selected techniques systematically and reference specific code files when relevant.
```

## Supported AI Assistants

- ✅ **Cursor IDE** (Primary support)
- ✅ **Claude Desktop** (with MCP configuration)
- ✅ **Cline** (VS Code extension)
- ✅ **Windsurf** (with MCP setup)
- ✅ **Any MCP-compatible client**

## Privacy & Security

- 🔒 **Local Code Analysis**: All workspace scanning happens on your machine
- 🌐 **API-Only Techniques**: Only technique selection uses external API
- 🔐 **No Code Upload**: Your code never leaves your computer
- 🎯 **Minimal Data**: Only question analysis sent to technique API

## Architecture

```
Your Question → Local Workspace Scan → Qdrant Cloud → Enhanced Prompt
     ↓              ↓ (Private)           ↓ (Vector Search)   ↓
  "Optimize     Finds relevant      Semantic similarity   Returns enhanced
   React app"   code files         technique selection   prompt with scores
```

## Troubleshooting

### Server Not Starting?
```bash
# Check if Python dependencies are installed
pip install fastmcp sentence-transformers httpx

# Verify API keys are set
echo $GROQ_API_KEY
echo $TAVILY_API_KEY  
echo $PROMPTGEN_API_KEY
```

### Tool Not Appearing in Cursor?
1. Verify `~/.cursor/mcp_servers.json` exists and has correct format
2. Restart Cursor completely (not just reload window)
3. Check Command Palette for "MCP" options
4. Look for MCP status in Cursor's status bar

### Getting API Errors?
- **PromptGen**: Confirm key at promptgen.dev - this is required for technique access
- **GROQ**: Verify API key at console.groq.com (optional - for LLM responses)
- **Tavily**: Check quota at tavily.com/dashboard (optional - for web search)

## Examples

### Debugging Help
**Input**: `"This function is throwing an error"`
**Enhancement**: Adds error analysis techniques, relevant code context, and systematic debugging steps

### Architecture Questions  
**Input**: `"How should I structure this feature?"`
**Enhancement**: Applies architectural thinking patterns, includes existing codebase patterns, and provides structured design approaches

### Performance Optimization
**Input**: `"Make this faster"`
**Enhancement**: Uses performance analysis techniques, identifies bottlenecks in your code, and suggests specific optimizations

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

This is a specialized MCP server for prompt enhancement. For issues or feature requests, please check the documentation or contact support through promptgen.dev.

---

**Ready to supercharge your AI interactions?** Get your API keys and start generating better prompts in minutes! 🚀 