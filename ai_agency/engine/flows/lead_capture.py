from ai_agency.engine.llm import call_llm

async def handle_turn(client_id: str, channel: str, session_id: str, message: str) -> str:
    """
    Basic slot-filling flow:
    1. Collect name, phone, email, service
    2. Confirm and book in calendar
    """
    # TODO: manage session state (slot filling)
    prompt = f"You are a receptionist for client {client_id}. The user said: {message}"
    reply = await call_llm(prompt)
    return reply
