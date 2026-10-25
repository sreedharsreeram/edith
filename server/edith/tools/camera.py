"""Tools that use the glasses' camera beyond the one photo taken when the button was pressed."""

from ..agent.harness import Tool, ToolContext, ToolResult
from ..vision import select_views

COUNTDOWN_S = 3
BURST_FPS = 2
MAX_VIEWS = 8  # enough angles to identify something, without paying for 30 images


async def walk_around(ctx: ToolContext, args: dict) -> ToolResult:
    frames = await ctx.device.capture(
        instruction=args.get("instruction", "Walk slowly around it"),
        countdown_s=COUNTDOWN_S,
        seconds=args.get("seconds", 12),
        fps=BURST_FPS,
    )
    views = select_views(frames, MAX_VIEWS)
    return ToolResult(
        f"The wearer walked around it. {len(frames)} frames captured; these are the {len(views)} sharpest, "
        "most different views, in the order they were taken.",
        images=views,
    )


async def take_photo(ctx: ToolContext, args: dict) -> ToolResult:
    frames = await ctx.device.capture(instruction=args["instruction"], countdown_s=COUNTDOWN_S, seconds=1, fps=BURST_FPS)
    return ToolResult("Here is the new photo.", images=select_views(frames, 1))


INSTRUCTION = {
    "type": "string",
    "minLength": 1,
    "maxLength": 40,  # has to fit the glasses' display
    "description": "Short instruction shown to the wearer, e.g. 'Walk slowly around the car'.",
}

CAMERA_TOOLS = [
    Tool(
        name="walk_around",
        description=(
            "Ask the wearer to walk slowly around the object while the glasses record. Returns several views "
            "from different angles. Use when one photo isn't enough to identify or describe it well."
        ),
        parameters={
            "type": "object",
            "properties": {
                "instruction": INSTRUCTION,
                "seconds": {"type": "integer", "minimum": 5, "maximum": 20, "description": "Recording length."},
            },
            "required": ["instruction"],
            "additionalProperties": False,
        },
        run=walk_around,
        timeout_s=60,
    ),
    Tool(
        name="take_photo",
        description=(
            "Ask the wearer to point the camera at something specific (an info plaque, a label, a screen) and "
            "take a new photo of it."
        ),
        parameters={
            "type": "object",
            "properties": {"instruction": INSTRUCTION},
            "required": ["instruction"],
            "additionalProperties": False,
        },
        run=take_photo,
        timeout_s=30,
    ),
]
