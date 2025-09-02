# Agent Architecture Documentation

This document provides detailed technical documentation for the AI Agency Boilerplate's agent architecture, covering the modular design, conversation flows, and channel adapters.

## 🏗️ Architecture Overview

The system follows a **Purpose × Interface** matrix design where:

- **Purposes** define what the agent does (Lead Capture, Support)
- **Interfaces** define how users interact (Phone, WhatsApp, Web)
- **Core Engine** handles conversation logic and tool orchestration
- **Channel Adapters** translate between interfaces and the core engine

```
┌─────────────────────────────────────────────────────────────┐
│                    Channel Adapters                         │
├─────────────┬─────────────┬─────────────┬───────────────────┤
│   Phone     │  WhatsApp   │   Web Chat  │   Future Channels │
│ (Twilio)    │ (Twilio)    │ (HTTP/WS)   │ (Email, Slack)    │
└─────────────┴─────────────┴─────────────┴───────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core Conversation Engine                 │
├─────────────────────────────────────────────────────────────┤
│  • Message Normalization  • Session Management             │
│  • Purpose Routing        • Context Management             │
│  • LLM Orchestration      • Tool Integration               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Purpose Flows                            │
├─────────────────────┬───────────────────────────────────────┤
│    Lead Capture     │              Support                  │
│  • Slot Filling     │         • Intent Classification      │
│  • Validation       │         • RAG Retrieval              │
│  • Booking Logic    │         • Confidence Scoring         │
│  • CRM Integration  │         • Escalation Logic           │
└─────────────────────┴───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Tools Layer                            │
├─────────────┬─────────────┬─────────────┬───────────────────┤
│  Calendar   │     CRM     │     RAG     │   Voice Services  │
│ (Google,    │ (HubSpot,   │ (Qdrant,    │ (STT, TTS)        │
│  Calendly)  │ Pipedrive)  │ pgvector)   │                   │
└─────────────┴─────────────┴─────────────┴───────────────────┘
```

## 🧠 Core Conversation Engine - 🚧 WIP

The conversation engine is the central orchestrator that handles all agent interactions regardless of channel or purpose.

### Engine Interface

```python
async def handle_turn(
    client_id: str,
    channel: str,
    purpose: str,
    session_id: str,
    message: str,
    context: dict = None
) -> ConversationResponse:
    """
    Process a single turn in the conversation.
    
    Args:
        client_id: Client identifier (e.g., "spa-del-sol")
        channel: Communication channel ("phone", "whatsapp", "web")
        purpose: Conversation purpose ("lead_capture", "support")
        session_id: Unique session identifier
        message: User input (text or normalized from voice)
        context: Additional context (caller_id, locale, etc.)
    
    Returns:
        ConversationResponse with text, actions, and metadata
    """
```

### Message Normalization

All incoming messages are normalized to a common format:

```python
class NormalizedMessage:
    text: str                    # Cleaned text content
    channel: str                 # Source channel
    session_id: str              # Session identifier
    user_id: Optional[str]       # User identifier (if available)
    metadata: dict               # Channel-specific metadata
    timestamp: datetime          # Message timestamp
    locale: str = "en-US"        # Detected or configured locale
```

### Session Management

Sessions maintain conversation state and context:

```python
class ConversationSession:
    session_id: str
    client_id: str
    channel: str
    purpose: str
    state: dict                  # Purpose-specific state
    context: dict                # Conversation context
    created_at: datetime
    last_activity: datetime
    turns: List[ConversationTurn]
```

## 🎯 Purpose Flows - 🚧 WIP

### Lead Capture Flow - 🚧 WIP

The lead capture flow implements a slot-filling state machine to collect customer information and book appointments.

#### State Machine

```python
class LeadCaptureStates(Enum):
    GREETING = "greeting"
    COLLECTING_NAME = "collecting_name"
    COLLECTING_PHONE = "collecting_phone"
    COLLECTING_EMAIL = "collecting_email"
    COLLECTING_SERVICE = "collecting_service"
    COLLECTING_TIME = "collecting_time"
    CONFIRMING = "confirming"
    BOOKING = "booking"
    COMPLETED = "completed"
    ESCALATED = "escalated"
```

#### Slot Filling Logic

```python
class LeadCaptureFlow:
    def __init__(self, client_config: ClientConfig):
        self.config = client_config
        self.required_fields = client_config.booking.rules.lead_fields
        self.state_machine = LeadCaptureStateMachine()
    
    async def handle_turn(self, session: ConversationSession, message: str) -> str:
        current_state = session.state.get("current_state", LeadCaptureStates.GREETING)
        
        # Extract information from message
        extracted_data = await self.extract_entities(message, current_state)
        
        # Update session state
        session.state.update(extracted_data)
        
        # Determine next action
        next_action = await self.determine_next_action(session, extracted_data)
        
        # Generate response
        response = await self.generate_response(session, next_action)
        
        return response
```

#### Field Validation

Each field has specific validation rules:

```python
class FieldValidator:
    @staticmethod
    def validate_phone(phone: str) -> ValidationResult:
        # International phone number validation
        pass
    
    @staticmethod
    def validate_email(email: str) -> ValidationResult:
        # Email format validation
        pass
    
    @staticmethod
    def validate_service(service: str, available_services: List[str]) -> ValidationResult:
        # Service selection validation
        pass
    
    @staticmethod
    def validate_time(time_str: str, business_hours: dict) -> ValidationResult:
        # Time slot validation against business hours
        pass
```

### Support Flow - 🚧 WIP

The support flow uses RAG (Retrieval Augmented Generation) to answer questions from a knowledge base.

#### Intent Classification

```python
class SupportIntentClassifier:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def classify_intent(self, message: str) -> IntentResult:
        prompt = f"""
        Classify the following customer message into one of these intents:
        - FAQ: General questions about services, hours, location
        - BOOKING: Questions about appointments or scheduling
        - COMPLAINT: Issues or complaints
        - ESCALATION: Complex issues requiring human help
        
        Message: {message}
        """
        
        response = await self.llm.generate(prompt)
        return self.parse_intent_response(response)
```

#### RAG Pipeline

```python
class RAGPipeline:
    def __init__(self, vector_store, llm_client):
        self.vector_store = vector_store
        self.llm = llm_client
    
    async def retrieve_and_generate(self, query: str, client_id: str) -> RAGResult:
        # 1. Retrieve relevant documents
        docs = await self.vector_store.similarity_search(
            query=query,
            collection=f"{client_id}-faq",
            top_k=5
        )
        
        # 2. Generate answer with citations
        context = "\n".join([doc.content for doc in docs])
        answer = await self.llm.generate_with_context(query, context)
        
        # 3. Calculate confidence score
        confidence = await self.calculate_confidence(query, answer, docs)
        
        return RAGResult(
            answer=answer,
            confidence=confidence,
            sources=docs,
            citations=self.extract_citations(answer, docs)
        )
```

#### Confidence Scoring

```python
class ConfidenceScorer:
    async def calculate_confidence(self, query: str, answer: str, sources: List[Document]) -> float:
        # Multiple confidence signals:
        # 1. Semantic similarity between query and sources
        # 2. Answer completeness (covers all aspects of query)
        # 3. Source quality and recency
        # 4. LLM self-assessment
        
        semantic_score = await self.calculate_semantic_similarity(query, sources)
        completeness_score = await self.assess_completeness(query, answer)
        source_quality = await self.assess_source_quality(sources)
        
        return (semantic_score * 0.4 + completeness_score * 0.4 + source_quality * 0.2)
```

## 📡 Channel Adapters - 🚧 WIP

### Phone Adapter (Twilio) - 🚧 WIP

The phone adapter handles real-time voice conversations using Twilio Media Streams.

#### Media Stream Handler

```python
class PhoneAdapter:
    def __init__(self, stt_client, tts_client, conversation_engine):
        self.stt = stt_client
        self.tts = tts_client
        self.engine = conversation_engine
    
    async def handle_media_stream(self, stream_sid: str, client_id: str):
        """Handle real-time audio stream from Twilio."""
        session_id = f"phone_{stream_sid}"
        audio_buffer = []
        
        async for media in self.stream_events:
            # Accumulate audio chunks
            audio_buffer.append(media.payload)
            
            # Process when we have enough audio or silence detected
            if self.should_process_audio(audio_buffer):
                # Convert audio to text
                text = await self.stt.transcribe(audio_buffer)
                
                if text.strip():
                    # Process with conversation engine
                    response = await self.engine.handle_turn(
                        client_id=client_id,
                        channel="phone",
                        purpose="lead_capture",  # or from config
                        session_id=session_id,
                        message=text
                    )
                    
                    # Convert response to speech
                    audio_response = await self.tts.synthesize(response.text)
                    
                    # Stream back to caller
                    await self.send_audio_to_twilio(stream_sid, audio_response)
                
                audio_buffer.clear()
```

#### Barge-in Support

```python
class BargeInHandler:
    async def handle_interruption(self, stream_sid: str, new_audio: bytes):
        """Handle user interrupting the bot's speech."""
        # Stop current TTS playback
        await self.stop_tts_playback(stream_sid)
        
        # Process the interruption
        text = await self.stt.transcribe(new_audio)
        # Continue with normal conversation flow
```

### WhatsApp Adapter - 🚧 WIP

The WhatsApp adapter handles text messages, media, and interactive elements.

#### Message Processing

```python
class WhatsAppAdapter:
    async def handle_webhook(self, payload: dict, client_id: str):
        """Process incoming WhatsApp webhook."""
        message = self.normalize_whatsapp_message(payload)
        
        response = await self.conversation_engine.handle_turn(
            client_id=client_id,
            channel="whatsapp",
            purpose=self.determine_purpose(message),
            session_id=message.session_id,
            message=message.text
        )
        
        # Send response back to WhatsApp
        await self.send_whatsapp_message(
            to=message.from_number,
            text=response.text,
            interactive_elements=response.interactive_elements
        )
```

#### Interactive Messages

```python
class InteractiveMessageBuilder:
    def build_quick_replies(self, options: List[str]) -> dict:
        """Build WhatsApp quick reply buttons."""
        return {
            "type": "interactive",
            "interactive": {
                "type": "button",
                "body": {"text": "Please select an option:"},
                "action": {
                    "buttons": [
                        {"id": f"option_{i}", "title": option}
                        for i, option in enumerate(options)
                    ]
                }
            }
        }
    
    def build_list_message(self, sections: List[dict]) -> dict:
        """Build WhatsApp list message."""
        return {
            "type": "interactive",
            "interactive": {
                "type": "list",
                "body": {"text": "Please select from the list:"},
                "action": {
                    "button": "View Options",
                    "sections": sections
                }
            }
        }
```

### Web Chat Adapter - 🚧 WIP

The web chat adapter provides real-time chat functionality with streaming responses.

#### WebSocket Handler

```python
class WebChatAdapter:
    async def handle_websocket(self, websocket: WebSocket, client_id: str):
        """Handle WebSocket connection for real-time chat."""
        await websocket.accept()
        session_id = f"web_{uuid.uuid4()}"
        
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_json()
                message = data.get("message", "")
                
                # Process with conversation engine
                response = await self.conversation_engine.handle_turn(
                    client_id=client_id,
                    channel="web",
                    purpose=data.get("purpose", "lead_capture"),
                    session_id=session_id,
                    message=message
                )
                
                # Stream response back to client
                await self.stream_response(websocket, response)
                
        except WebSocketDisconnect:
            await self.cleanup_session(session_id)
```

#### Streaming Response

```python
class ResponseStreamer:
    async def stream_response(self, websocket: WebSocket, response: ConversationResponse):
        """Stream response tokens to client for real-time typing effect."""
        if response.text:
            # Stream text token by token
            for token in self.tokenize_response(response.text):
                await websocket.send_json({
                    "type": "token",
                    "content": token
                })
                await asyncio.sleep(0.05)  # Typing delay
        
        # Send final response with metadata
        await websocket.send_json({
            "type": "complete",
            "response": response.text,
            "actions": response.actions,
            "metadata": response.metadata
        })
```

## 🔧 Tools Integration - 🚧 WIP

### Calendar Integration - 🚧 WIP

```python
class CalendarTool:
    def __init__(self, google_calendar_client, calendly_client):
        self.google_cal = google_calendar_client
        self.calendly = calendly_client
    
    async def check_availability(self, client_id: str, date_range: tuple, service_duration: int) -> List[TimeSlot]:
        """Check available time slots for booking."""
        config = load_client_config(client_id)
        
        if config.booking.provider == "google_calendar":
            return await self.google_cal.get_available_slots(
                calendar_id=config.booking.calendar_id,
                start_time=date_range[0],
                end_time=date_range[1],
                duration_minutes=service_duration
            )
        elif config.booking.provider == "calendly":
            return await self.calendly.get_availability(
                event_type=config.booking.event_type,
                start_time=date_range[0],
                end_time=date_range[1]
            )
    
    async def create_booking(self, client_id: str, booking_data: dict) -> BookingResult:
        """Create a new booking."""
        config = load_client_config(client_id)
        
        # Validate business hours
        if not self.is_within_business_hours(booking_data["datetime"], config.business_hours):
            return BookingResult(success=False, error="Outside business hours")
        
        # Create booking
        if config.booking.provider == "google_calendar":
            event = await self.google_cal.create_event(
                calendar_id=config.booking.calendar_id,
                title=f"{booking_data['service']} - {booking_data['name']}",
                start_time=booking_data["datetime"],
                duration_minutes=booking_data["duration"],
                attendees=[booking_data["email"]],
                description=self.format_booking_description(booking_data)
            )
            return BookingResult(success=True, booking_id=event.id)
```

### CRM Integration - 🚧 WIP

```python
class CRMTool:
    def __init__(self, hubspot_client, pipedrive_client, airtable_client):
        self.hubspot = hubspot_client
        self.pipedrive = pipedrive_client
        self.airtable = airtable_client
    
    async def create_lead(self, client_id: str, lead_data: dict) -> LeadResult:
        """Create a new lead in the configured CRM."""
        config = load_client_config(client_id)
        
        # Map lead data according to config
        mapped_data = self.map_lead_data(lead_data, config.crm.map)
        
        if config.crm.provider == "hubspot":
            contact = await self.hubspot.create_contact(mapped_data)
            return LeadResult(success=True, lead_id=contact.id)
        
        elif config.crm.provider == "pipedrive":
            person = await self.pipedrive.create_person(mapped_data)
            return LeadResult(success=True, lead_id=person.id)
        
        elif config.crm.provider == "airtable":
            record = await self.airtable.create_record(
                base_id=config.crm.base_id,
                table_name=config.crm.table_name,
                fields=mapped_data
            )
            return LeadResult(success=True, lead_id=record.id)
```

### RAG Integration - 🚧 WIP

```python
class RAGTool:
    def __init__(self, vector_store, embedding_client):
        self.vector_store = vector_store
        self.embeddings = embedding_client
    
    async def ingest_documents(self, client_id: str, documents: List[Document]) -> IngestResult:
        """Ingest documents into the vector store."""
        collection_name = f"{client_id}-faq"
        
        # Create collection if it doesn't exist
        await self.vector_store.create_collection(collection_name)
        
        # Process documents
        processed_docs = []
        for doc in documents:
            # Chunk document
            chunks = self.chunk_document(doc)
            
            # Generate embeddings
            for chunk in chunks:
                embedding = await self.embeddings.embed(chunk.text)
                chunk.embedding = embedding
                processed_docs.append(chunk)
        
        # Store in vector database
        await self.vector_store.upsert(collection_name, processed_docs)
        
        return IngestResult(success=True, documents_processed=len(processed_docs))
    
    async def search(self, client_id: str, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for relevant documents."""
        collection_name = f"{client_id}-faq"
        
        # Generate query embedding
        query_embedding = await self.embeddings.embed(query)
        
        # Search vector store
        results = await self.vector_store.similarity_search(
            collection=collection_name,
            query_vector=query_embedding,
            top_k=top_k
        )
        
        return [SearchResult(doc=result.doc, score=result.score) for result in results]
```

## 🔄 State Management - 🚧 WIP

### Session State - 🚧 WIP

```python
class SessionStateManager:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def get_session_state(self, session_id: str) -> dict:
        """Retrieve session state from Redis."""
        key = f"session:{session_id}"
        data = await self.redis.get(key)
        return json.loads(data) if data else {}
    
    async def update_session_state(self, session_id: str, updates: dict):
        """Update session state in Redis."""
        key = f"session:{session_id}"
        current_state = await self.get_session_state(session_id)
        current_state.update(updates)
        
        # Set expiration (24 hours)
        await self.redis.setex(key, 86400, json.dumps(current_state))
    
    async def clear_session_state(self, session_id: str):
        """Clear session state."""
        key = f"session:{session_id}"
        await self.redis.delete(key)
```

### Conversation Context - 🚧 WIP

```python
class ConversationContext:
    def __init__(self, session_id: str, client_id: str):
        self.session_id = session_id
        self.client_id = client_id
        self.turns = []
        self.extracted_entities = {}
        self.user_preferences = {}
        self.conversation_metadata = {}
    
    def add_turn(self, user_message: str, bot_response: str, metadata: dict = None):
        """Add a conversation turn to context."""
        turn = ConversationTurn(
            user_message=user_message,
            bot_response=bot_response,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        self.turns.append(turn)
    
    def get_conversation_summary(self) -> str:
        """Generate a summary of the conversation."""
        if not self.turns:
            return "No conversation history."
        
        # Use LLM to summarize conversation
        conversation_text = "\n".join([
            f"User: {turn.user_message}\nBot: {turn.bot_response}"
            for turn in self.turns
        ])
        
        return self.summarize_conversation(conversation_text)
```

## 🚨 Error Handling & Escalation - 🚧 WIP

### Error Classification - 🚧 WIP

```python
class ErrorClassifier:
    def classify_error(self, error: Exception, context: dict) -> ErrorType:
        """Classify errors for appropriate handling."""
        if isinstance(error, ValidationError):
            return ErrorType.VALIDATION_ERROR
        elif isinstance(error, IntegrationError):
            return ErrorType.INTEGRATION_ERROR
        elif isinstance(error, LLMError):
            return ErrorType.LLM_ERROR
        else:
            return ErrorType.UNKNOWN_ERROR
```

### Escalation Logic - 🚧 WIP

```python
class EscalationManager:
    async def should_escalate(self, session: ConversationSession, error: Exception) -> bool:
        """Determine if conversation should be escalated to human."""
        # Escalate if:
        # 1. Multiple consecutive errors
        # 2. User explicitly requests human
        # 3. Low confidence in responses
        # 4. Complex issue beyond bot capabilities
        
        error_count = session.metadata.get("error_count", 0)
        if error_count >= 3:
            return True
        
        if "human" in session.last_user_message.lower():
            return True
        
        return False
    
    async def escalate_to_human(self, session: ConversationSession, reason: str):
        """Escalate conversation to human agent."""
        config = load_client_config(session.client_id)
        
        # Send notification to human agents
        if config.escalation.to_slack_webhook:
            await self.send_slack_notification(session, reason)
        
        if config.escalation.to_email:
            await self.send_email_notification(session, reason)
        
        # Update session state
        session.state["escalated"] = True
        session.state["escalation_reason"] = reason
        session.state["escalation_time"] = datetime.utcnow()
```

## 📊 Monitoring & Analytics - 🚧 WIP

### Conversation Metrics - 🚧 WIP

```python
class ConversationMetrics:
    def __init__(self, metrics_client):
        self.metrics = metrics_client
    
    async def record_turn(self, session: ConversationSession, turn_data: dict):
        """Record conversation turn metrics."""
        await self.metrics.increment("conversation.turns", tags={
            "client_id": session.client_id,
            "channel": session.channel,
            "purpose": session.purpose
        })
        
        await self.metrics.timing("conversation.response_time", turn_data["response_time_ms"], tags={
            "client_id": session.client_id,
            "channel": session.channel
        })
    
    async def record_completion(self, session: ConversationSession, success: bool):
        """Record conversation completion metrics."""
        await self.metrics.increment("conversation.completions", tags={
            "client_id": session.client_id,
            "channel": session.channel,
            "purpose": session.purpose,
            "success": str(success)
        })
```

### Cost Tracking - 🚧 WIP

```python
class CostTracker:
    def __init__(self, metrics_client):
        self.metrics = metrics_client
    
    async def track_llm_usage(self, client_id: str, model: str, tokens: int, cost: float):
        """Track LLM usage and costs."""
        await self.metrics.increment("llm.tokens", tokens, tags={
            "client_id": client_id,
            "model": model
        })
        
        await self.metrics.increment("llm.cost", cost, tags={
            "client_id": client_id,
            "model": model
        })
```

This architecture provides a robust, scalable foundation for building AI-powered business bots that can be quickly deployed and configured for different clients and use cases.
