"""Collection classes for managing multiple tools."""

from typing import Any

from anthropic.types.beta import BetaToolUnionParam

from .base import (
    BaseAnthropicTool,
    ToolError,
    ToolFailure,
    ToolResult,
)


class ToolCollection:
    """Enhanced tool collection with neural processing capabilities."""

    def __init__(self, *tools: BaseAnthropicTool, enable_neural: bool = True, api_key: str = None):
        """Initialize with neural enhancement capabilities"""
        from .neural import NeuralEnhancedComputerTool
        
        # Convert ComputerTool to NeuralEnhancedComputerTool if neural is enabled
        processed_tools = []
        for tool in tools:
            if enable_neural and api_key and tool.to_params()["name"] == "computer":
                tool = NeuralEnhancedComputerTool(api_key)
            processed_tools.append(tool)
            
        self.tools = processed_tools
        self.tool_map = {tool.to_params()["name"]: tool for tool in processed_tools}
        self.enable_neural = enable_neural

    def to_params(
        self,
    ) -> list[BetaToolUnionParam]:
        return [tool.to_params() for tool in self.tools]

    async def run(self, *, name: str, tool_input: dict[str, Any]) -> ToolResult:
        tool = self.tool_map.get(name)
        if not tool:
            return ToolFailure(error=f"Tool {name} is invalid")
        try:
            return await tool(**tool_input)
        except ToolError as e:
            return ToolFailure(error=e.message)
