import os
import logging
import json
import asyncio

from dotenv import load_dotenv

from agentmail import AgentMail
from websockets.asyncio.client import connect

from livekit import agents
from livekit.agents import (
    AgentSession,
    Agent,
    RoomInputOptions,
    JobContext,
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
    def __init__(self) -> None:
        super().__init__(
            instructions=f"You are a helpful voice and email AI assistant. You can send, receive, and reply to emails. Your email address is {inbox.inbox_id}.",
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
