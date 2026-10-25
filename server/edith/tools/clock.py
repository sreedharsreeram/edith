from datetime import datetime

from ..agent.harness import Tool, ToolContext


async def get_time(ctx: ToolContext, args: dict) -> str:
    now = datetime.now().astimezone()
    return f"{now:%A %d %B %Y, %H:%M} ({now.tzname()})"


CLOCK_TOOLS = [
    Tool(
        name="get_time",
        description="The current local date and time. Use for any question about today, the time, or 'how long until'.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        run=get_time,
    ),
]
