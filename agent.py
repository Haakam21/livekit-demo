import logging
import asyncio

from agentmail import AgentMail, AsyncAgentMail, Subscribe, MessageReceived
from agentmail_toolkit.livekit import AgentMailToolkit

from livekit import agents
from livekit.agents import (
    AgentSession,
    Agent,
    RoomInputOptions,
    JobContext,
    BackgroundAudioPlayer,
    AudioConfig,
    BuiltinAudioClip,
)
from livekit.plugins import openai, noise_cancellation, silero


logger = logging.getLogger(__name__)


class EmailAssistant(Agent):
    inbox_id: str
    ws_task: asyncio.Task | None = None

    def __init__(self) -> None:
        client = AgentMail()

        inbox = client.inboxes.create(
            username="lisa",
            display_name="Lisa",
            client_id="lisa-inbox",
        )

        self.inbox_id = inbox.inbox_id

        super().__init__(
            instructions=f"""
            You are a helpful voice and email AI assistant. Your name is Lisa. You can recieve emails at {self.inbox_id}. You can also send and reply to emails.
            When using email tools, use "{self.inbox_id}" as the inbox_id parameter. When writing emails, include "Lisa" in the signature.
            Always speak in English.
            IMPORTANT: {self.inbox_id} is your inbox, not the user's inbox.
            """,
            tools=AgentMailToolkit(client=client).get_tools(
                [
                    "list_threads",
                    "get_thread",
                    "get_attachment",
                    "send_message",
                    "reply_to_message",
                ]
            ),
        )

    async def _websocket_task(self):
        try:
            async with AsyncAgentMail().websockets.connect() as socket:
                logger.info("Connected to AgentMail websocket")

                await socket.send_subscribe(Subscribe(inbox_ids=[self.inbox_id]))

                while True:
                    data = await socket.recv()
                    logger.debug("Received data: %s", data)

                    if isinstance(data, MessageReceived):
                        self.session.interrupt()

                        await self.session.generate_reply(
                            instructions=f"""Say "I've recieved an email" and then read the email.""",
                            user_input=data.message.model_dump_json(),
                        )
        finally:
            logger.info("Disconnected from AgentMail websocket")

    async def on_enter(self):
        self.ws_task = asyncio.create_task(self._websocket_task())

        await self.session.generate_reply(
            instructions=f"In English, greet the user, introduce yourself as Lisa, inform them that you can recieve emails at {self.inbox_id}, and offer your assistance.",
            allow_interruptions=False,
        )

    async def on_exit(self):
        if self.ws_task:
            self.ws_task.cancel()


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
        llm=openai.realtime.RealtimeModel(voice="shimmer", turn_detection=None),
    )

    await session.start(
        room=ctx.room,
        agent=EmailAssistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
        ),
    )

    await BackgroundAudioPlayer(
        thinking_sound=[AudioConfig(BuiltinAudioClip.KEYBOARD_TYPING, volume=0.8)]
    ).start(room=ctx.room, agent_session=session)

    await ctx.connect()


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
