#!/bin/bash

echo "🚀 Setting up PromptGen MCP (Self-Contained)..."

# Check if we're in the right directory
if [ ! -f "src/prompt_gen_mcp/server.py" ]; then
    echo "❌ Please run this from the prompt-gen-mcp directory"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install fastmcp sentence-transformers httpx groq qdrant-client

# Check for API keys
if [ -z "$GROQ_API_KEY" ]; then
    echo "⚠️  Please set GROQ_API_KEY environment variable"
    echo "   Get your key from: https://console.groq.com/"
fi

# Test the MCP server
echo "🧪 Testing MCP server..."
if python -c "
import sys
sys.path.insert(0, 'src')
from prompt_gen_mcp.server import mcp
print('✅ MCP server imports successfully')
" 2>/dev/null; then
    echo "✅ MCP server is ready"
else
    echo "❌ MCP server has import issues"
    echo "   Make sure all dependencies are installed"
fi

# Generate Cursor config
CURRENT_DIR=$(pwd)
cat > cursor_config.json << EOF
{
  "mcpServers": {
    "prompt-gen": {
      "command": "python",
      "args": ["$CURRENT_DIR/src/prompt_gen_mcp/server.py"],
      "env": {
        "GROQ_API_KEY": "$GROQ_API_KEY"
      }
    }
  }
}
EOF

echo "📋 Cursor config generated: cursor_config.json"
echo "   Copy this to ~/.cursor/mcp_servers.json"
echo ""
echo "🎉 Setup complete! Next steps:"
echo "   1. Copy cursor_config.json to ~/.cursor/mcp_servers.json"
echo "   2. Restart Cursor completely"
echo "   3. Use Cmd+Shift+P → 'MCP: enhance_prompt'"
echo ""
echo "✨ This MCP server is self-contained - no background services needed!"
echo "🔒 All code scanning happens locally, only vectors sent to Qdrant" 

echo "🚀 Setting up PromptGen MCP (Self-Contained)..."

# Check if we're in the right directory
if [ ! -f "src/prompt_gen_mcp/server.py" ]; then
    echo "❌ Please run this from the prompt-gen-mcp directory"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install fastmcp sentence-transformers httpx groq qdrant-client

# Check for API keys
if [ -z "$GROQ_API_KEY" ]; then
    echo "⚠️  Please set GROQ_API_KEY environment variable"
    echo "   Get your key from: https://console.groq.com/"
fi

# Test the MCP server
echo "🧪 Testing MCP server..."
if python -c "
import sys
sys.path.insert(0, 'src')
from prompt_gen_mcp.server import mcp
print('✅ MCP server imports successfully')
" 2>/dev/null; then
    echo "✅ MCP server is ready"
else
    echo "❌ MCP server has import issues"
    echo "   Make sure all dependencies are installed"
fi

# Generate Cursor config
CURRENT_DIR=$(pwd)
cat > cursor_config.json << EOF
{
  "mcpServers": {
    "prompt-gen": {
      "command": "python",
      "args": ["$CURRENT_DIR/src/prompt_gen_mcp/server.py"],
      "env": {
        "GROQ_API_KEY": "$GROQ_API_KEY"
      }
    }
  }
}
EOF

echo "📋 Cursor config generated: cursor_config.json"
echo "   Copy this to ~/.cursor/mcp_servers.json"
echo ""
echo "🎉 Setup complete! Next steps:"
echo "   1. Copy cursor_config.json to ~/.cursor/mcp_servers.json"
echo "   2. Restart Cursor completely"
echo "   3. Use Cmd+Shift+P → 'MCP: enhance_prompt'"
echo ""
echo "✨ This MCP server is self-contained - no background services needed!"
echo "🔒 All code scanning happens locally, only vectors sent to Qdrant" 
 
 
 