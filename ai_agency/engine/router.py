from ai_agency.engine.flows import lead_capture, support
from ai_agency.engine.llm import call_llm

async def handle_message(client_id: str, channel: str, purpose: str, session_id: str, message: str):
    if purpose == "lead_capture":
        return await lead_capture.handle_turn(client_id, channel, session_id, message)
    elif purpose == "support":
        return await support.handle_turn(client_id, channel, session_id, message)
    else:
        return "Sorry, I don’t know how to handle this purpose yet."
