# AI Agency Boilerplate

A lean, production-ready boilerplate for building AI-powered business bots. Deploy AI receptionists, lead capture bots, and support assistants without the complexity of multi-tenant SaaS platforms.

## 🎯 What This Is

This is a **modular boilerplate** designed for:

- **Developers** building AI-powered customer interaction for their own websites
- **SaaS builders** creating AI bot services
- **Agencies** deploying AI solutions for clients
- **Businesses** wanting AI-powered customer interaction (SPAs, dentists, gyms, real estate, etc.)

The system provides a clean API with conversation flows, channel adapters, and tool integrations that you can customize for your specific needs.

## 🏗️ Architecture Overview

```
[Phone]   [WhatsApp]   [Web Widget]
   │           │            │
   └─► Channel Adapters (webhooks, media streams, auth validation)
                │
           Core Service ──► LLM Engine ──► Tools (Calendar/CRM/RAG)
                │                 │
          Transcript DB      Vector DB / Docs store
                │
         Worker Queue (async) ──► CRM writes, indexing, summaries
```

### Core Components

- **Core Service**: Stateless FastAPI backend with conversation engine
- **Channel Adapters**: Thin adapters for Phone (Twilio), WhatsApp/SMS, Web widget
- **Purpose Flows**: Pluggable flows for Lead Capture and Support
- **Tools**: Shared capabilities (Calendar, CRM, RAG, STT/TTS)
- **Worker Tier**: Async tasks for CRM pushes, doc indexing, summaries
- **Configuration**: Per-client YAML configs (no UI needed)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo>
cd ai_agency_starter
pip install -r requirements.txt
```

### 2. Environment Configuration

Create `.env` file:

```bash
# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Twilio (for Phone/WhatsApp)
TWILIO_AUTH_TOKEN=...
TWILIO_ACCOUNT_SID=...

# Voice Services
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=...

# Integrations
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
HUBSPOT_API_KEY=...

# Infrastructure
DATABASE_URL=postgres://...
REDIS_URL=redis://...
QDRANT_URL=...
QDRANT_API_KEY=...

# Security
SERVER_API_KEY=your-super-secret-api-key
```

### 3. Client Configuration

Create a client config in `configs/`:

```yaml
# configs/spa-del-sol.yaml
client_id: "spa-del-sol"
business_profile:
  name: "SPA del Sol"
  location: "Barcelona"
  phone: "+34..."
  services:
    - name: "Massage 60m"
      duration_min: 60
      calendar: "cal_spa_general"
  business_hours:
    mon_fri: "10:00-19:00"
    sat: "10:00-14:00"

booking:
  provider: "google_calendar"
  rules:
    lead_fields: ["name","phone","email","service","preferred_time"]
    double_opt_in_sms: false

crm:
  provider: "hubspot"
  pipeline: "Leads"
  map:
    name: "$lead.name"
    phone: "$lead.phone"
    email: "$lead.email"
    notes: "$conversation.summary"

channels:
  phone:
    enabled: true
    stt: "whisper"
    tts: "elevenlabs"
  whatsapp:
    enabled: true
  web:
    enabled: true

support:
  rag:
    provider: "qdrant"
    collection: "spa-del-sol-faq"
    fallback_to_handoff: true

escalation:
  to_email: "reception@spadelsol.com"
  to_slack_webhook: "https://hooks.slack.com/..."

policies:
  pii_redaction: true
  profanity_filter: true
```

### 4. Run the Service

```bash
# Development
uvicorn ai_agency.main:app --reload

# Production
uvicorn ai_agency.main:app --host 0.0.0.0 --port 8000
```

## 📱 Supported Channels

### Phone (AI Receptionist) - 🚧 WIP
- **Provider**: Twilio with Media Streams
- **Features**: Real-time STT/TTS, barge-in support, DTMF
- **Use Case**: Answer calls, book appointments, collect leads
- **Endpoint**: `POST /phone/webhook`

### WhatsApp/SMS - 🚧 WIP
- **Provider**: Twilio WhatsApp Business API
- **Features**: Interactive messages, media support, quick replies
- **Use Case**: Lead capture, support, appointment reminders
- **Endpoint**: `POST /whatsapp/webhook`

### Web Chat Widget - 🚧 WIP
- **Provider**: Custom React widget or embeddable iframe
- **Features**: Streaming responses, session management, typing indicators
- **Use Case**: Website lead capture, support, FAQ
- **Endpoint**: `POST /web/chat`

## 🎯 Purpose Flows

### Lead Capture - 🚧 WIP
Collects customer information and books appointments:

1. **Greet** → Welcome and identify intent
2. **Collect Fields** → Name, phone, email, service, preferred time
3. **Validate** → Check business hours and availability
4. **Confirm** → Present options and get confirmation
5. **Book** → Create calendar event
6. **CRM Push** → Send lead data to CRM (async)
7. **Thank You** → Confirmation and next steps

### Support - 🚧 WIP
Answers FAQs and escalates when needed:

1. **Classify Intent** → Determine if it's a support question
2. **Retrieve** → Search knowledge base using RAG
3. **Synthesize** → Generate answer with citations
4. **Confidence Check** → If low confidence, ask clarifying questions
5. **Escalate** → Hand off to human if needed
6. **Follow Up** → Continue conversation or close

## 🔧 Tools & Integrations

### Calendar Integration - 🚧 WIP
- **Google Calendar**: Service account authentication, availability checking, event creation
- **Calendly**: Availability API, booking creation
- **Features**: Business hours validation, conflict detection, timezone handling

### CRM Integration - 🚧 WIP
- **HubSpot**: Contact creation, deal tracking, custom properties
- **Pipedrive**: Lead management, pipeline stages
- **Airtable**: Flexible data storage, custom fields
- **Features**: Retry logic, dead letter queues, field mapping

### RAG (Retrieval Augmented Generation) - 🚧 WIP
- **Vector Store**: Qdrant or pgvector for embeddings
- **Document Sources**: Notion, Confluence, Google Docs, PDFs
- **Features**: Hybrid search, citation tracking, confidence scoring

### Voice Services - 🚧 WIP
- **STT**: OpenAI Whisper, AssemblyAI
- **TTS**: ElevenLabs, OpenAI TTS
- **Features**: Streaming, voice cloning, language detection

## 🚀 Deployment Options

### Local Development - ✅ Basic Setup
```bash
# Using Docker Compose - 🚧 WIP
docker-compose up -d

# Or run directly
uvicorn ai_agency.main:app --reload
```

### Production Deployments - 🚧 WIP

#### Railway - 🚧 WIP
```bash
railway login
railway init
railway up
```

#### Fly.io - 🚧 WIP
```bash
fly launch
fly deploy
```

#### AWS (ECS/Fargate) - 🚧 WIP
```bash
# Use provided Terraform configs
cd deploy/terraform
terraform init
terraform apply
```

#### Vercel - 🚧 WIP
```bash
vercel --prod
```

## 📊 Monitoring & Observability - 🚧 WIP

### Logging - 🚧 WIP
- Structured JSON logs with client_id, channel, purpose, trace_id
- PII redaction for sensitive data
- Conversation transcripts stored for QA

### Metrics - 🚧 WIP
- Token usage and cost tracking per conversation
- Response time monitoring per channel
- Success/failure rates for integrations

### Alerts - 🚧 WIP
- Failed webhook deliveries
- High error rates
- Cost threshold breaches

## 🔐 Security - 🚧 WIP

### Authentication - 🚧 WIP
- Twilio webhook signature validation
- API key authentication for internal endpoints
- Per-client secrets via environment variables

### Privacy - 🚧 WIP
- PII redaction in logs
- Encrypted storage for sensitive data
- GDPR compliance features

### Rate Limiting - 🚧 WIP
- Per-client rate limits using Redis
- Channel-specific throttling
- Abuse prevention

## 📚 Usage Examples - 🚧 WIP

### Setting Up a New Client - 🚧 WIP

1. **Create Config**: Add `configs/client-name.yaml`
2. **Deploy**: Use CLI or manual deployment
3. **Test**: Verify webhook endpoints
4. **Go Live**: Update client's Twilio/website settings

### CLI Commands - 🚧 WIP

```bash
# Create new client from template
ai-agency new-client spa-del-sol --template lead-support-es

# Ingest documents for RAG
ai-agency ingest --client spa-del-sol ./docs/

# Rotate secrets
ai-agency rotate-secrets --client spa-del-sol

# Deploy to production
ai-agency deploy --client spa-del-sol --env production
```

## 🧪 Testing - 🚧 WIP

### Unit Tests - 🚧 WIP
```bash
pytest tests/unit/
```

### Integration Tests - 🚧 WIP
```bash
pytest tests/integration/
```

### Contract Tests - 🚧 WIP
```bash
pytest tests/contracts/
```

## 📖 Documentation

- [Agent Architecture](agents.md) - Detailed technical documentation
- [Configuration Guide](docs/configuration.md) - 🚧 WIP Client config reference
- [Deployment Guide](docs/deployment.md) - 🚧 WIP Production deployment steps
- [Integration Guide](docs/integrations.md) - 🚧 WIP Setting up external services

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: GitHub Issues for bugs and feature requests
- **Discussions**: GitHub Discussions for questions and ideas

---

**Built for developers who want to add AI-powered customer interaction to their applications without building everything from scratch.**
