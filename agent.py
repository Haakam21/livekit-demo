import os
import logging
import json
import asyncio

from dotenv import load_dotenv

from agentmail import AgentMail
from agentmail_toolkit import AgentMailToolkit, Tool
from websockets.asyncio.client import connect

from livekit import agents
from livekit.agents import (
    AgentSession,
    Agent,
    RoomInputOptions,
    JobContext,
    RunContext,
    function_tool,
)
from livekit.plugins import (
    openai,
    noise_cancellation,
)

load_dotenv()

api_key = os.getenv("AGENTMAIL_API_KEY")
ws_url = f"wss://ws.agentmail.to/v0?auth_token={api_key}"


logger = logging.getLogger(__name__)

client = AgentMail()

inbox = client.inboxes.create(
    username="livekit",
    display_name="LiveKit",
    client_id="livekit-inbox",
)


class Assistant(Agent):
    def _build_tool(self, tool: Tool):
        async def fn(raw_arguments: dict[str, object], context: RunContext):
            return tool.fn(**raw_arguments).json()

        return function_tool(
            f=fn,
            name=tool.name,
            description=tool.description,
            raw_schema={
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.schema.model_json_schema(),
            },
        )

    def __init__(self) -> None:
        super().__init__(
            instructions=f"You are a helpful voice and email AI assistant. You can send, receive, and reply to emails. Your email address is {inbox.inbox_id}. IMPORTANT: When using email tools, always use '{inbox.inbox_id}' as the inbox_id parameter.",
            tools=[
                self._build_tool(tool)
                for tool in AgentMailToolkit(client=client).get_tools(
                    ["list_threads", "get_thread", "send_message", "reply_to_message"]
                )
            ],
        )


async def websocket_task(inbox_id, session):
    """Background task to handle websocket connection"""
    try:
        async with connect(ws_url) as ws:
            logger.info("Connected to websocket server")

            await ws.send(json.dumps({"type": "subscribe", "inbox_ids": [inbox_id]}))

            while True:
                data = json.loads(await ws.recv())
                logger.debug("Received data: %s", data)

                if data["type"] == "event" and data["event_type"] == "message.received":
                    await session.generate_reply(
                        instructions=f""""Say "I've recieved an email" and then read the email.""",
                        user_input=str(data["message"]),
                    )
    except Exception as e:
        logger.error("Error: %s", e)


async def entrypoint(ctx: JobContext):
    session = AgentSession(llm=openai.realtime.RealtimeModel(voice="coral"))

    asyncio.create_task(websocket_task(inbox.inbox_id, session))

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

    await session.generate_reply(
        instructions=f"In English, greet the user, inform them that you can recieve emails at {inbox.inbox_id}, and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
