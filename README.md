# PromptGen MCP Server

A **Model Context Protocol (MCP)** server that enhances your prompts with 47+ advanced prompt engineering techniques, automatically selecting the best techniques based on your question type and providing relevant code context from your workspace.

## What is MCP?

**Model Context Protocol (MCP)** is an open standard that allows AI assistants like Claude, Cursor, and others to connect with external tools and data sources. Think of it as a way to give your AI assistant superpowers by connecting it to specialized services.

This MCP server acts as a "prompt enhancement engine" that:
- 🧠 **Analyzes your questions** to understand what you're trying to accomplish
- 🔍 **Scans your workspace** to find relevant code context
- 🎯 **Selects optimal techniques** from 47+ prompt engineering methods
- ✨ **Generates enhanced prompts** that get better AI responses

## Features

### 🚀 **Intelligent Technique Selection**
Automatically chooses from 47+ prompt engineering techniques including:
- **Chain of Thought** - Step-by-step reasoning
- **Few-Shot Learning** - Learning from examples
- **Plan and Solve** - Structured problem solving
- **Self-Ask** - Breaking down complex questions
- **Tree of Thought** - Exploring multiple solution paths
- **And 42 more advanced techniques...**

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

### 1. Get API Keys (Free Tiers Available)
```bash
# Required API keys:
GROQ_API_KEY=your_groq_key_here          # Free: 6,000 requests/day
TAVILY_API_KEY=your_tavily_key_here      # Free: 1,000 searches/month
PROMPTGEN_API_KEY=your_promptgen_key     # Get from promptgen.dev/api
```

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
# Code Context Analysis
Based on your workspace scan, I found these relevant files:
- src/components/UserProfile.tsx (lines 45-89)
- src/hooks/useUserData.ts (lines 12-34)

# Selected Prompt Engineering Techniques

## Chain of Thought
Let's approach this optimization systematically:
1. First, analyze current performance bottlenecks
2. Then, identify optimization opportunities
3. Finally, implement improvements step by step

## Plan and Solve
**Plan**: Break down the optimization into measurable steps
**Solve**: Apply specific React optimization patterns

# Enhanced Question
Given the React component in src/components/UserProfile.tsx that handles user data fetching and rendering, how can I optimize its performance considering:

1. **Rendering efficiency**: Are there unnecessary re-renders?
2. **Data fetching**: Can we improve the useUserData hook?
3. **Memory usage**: Are there potential memory leaks?
4. **Bundle size**: Can we reduce the component's footprint?

Please provide specific code improvements with before/after examples.
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
Your Question → Local Workspace Scan → Technique API → Enhanced Prompt
     ↓              ↓ (Private)           ↓ (Public)        ↓
  "Optimize     Finds relevant      Selects optimal    Returns enhanced
   React app"   code files         techniques         prompt with context
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
- **GROQ**: Verify API key at console.groq.com
- **Tavily**: Check quota at tavily.com/dashboard  
- **PromptGen**: Confirm key at promptgen.dev/api

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