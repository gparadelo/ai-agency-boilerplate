from fastapi import FastAPI
from ai_agency.routes import phone_twilio, whatsapp_twilio, web_chat

app = FastAPI(title="AI Agency Boilerplate")

# Include channel routes
app.include_router(phone_twilio.router, prefix="/phone", tags=["phone"])
app.include_router(whatsapp_twilio.router, prefix="/whatsapp", tags=["whatsapp"])
app.include_router(web_chat.router, prefix="/web", tags=["web"])

@app.get("/health")
def health_check():
    return {"status": "ok"}
