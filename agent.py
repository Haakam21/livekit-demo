from dotenv import load_dotenv

from agentmail import AgentMail

from livekit import agents
from livekit.agents import AgentSession, Agent, ChatContext, RoomInputOptions
from livekit.plugins import (
    openai,
    noise_cancellation,
)

load_dotenv()

client = AgentMail()

inbox = client.inboxes.create(
    username="livekit",
    display_name="LiveKit",
    client_id="livekit-inbox",
)


class Assistant(Agent):
    def __init__(self) -> None:
        threads = client.inboxes.threads.list(inbox_id=inbox.inbox_id, limit=20)

        initial_ctx = ChatContext()
        initial_ctx.add_message(
            role="system",
            content=f"Here are the latest threads in your inbox: {str(threads)}",
        )

        super().__init__(
            instructions=f"You are a helpful voice and email AI assistant. Your email address is {inbox.inbox_id}.",
            chat_ctx=initial_ctx,
        )


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(llm=openai.realtime.RealtimeModel(voice="coral"))

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

    await session.generate_reply(
        instructions="Greet the user, summarize the latest threads in your inbox, and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
