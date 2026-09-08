# Portfolio AI Voice Agent — Sada

A real-time, browser-based AI voice assistant built for a personal portfolio. **Sada** lets visitors speak naturally, see their own speech appear live, receive grounded answers about Vishal Sahil, and hear the AI response while the response text appears in sync with the generated voice.

The system combines **FastAPI**, **WebSockets**, **Deepgram streaming STT**, **Pinecone RAG**, **Gemini streaming**, **MongoDB conversation memory**, and **ElevenLabs streaming TTS** in one low-latency voice pipeline.

> Repository: `https://github.com/vishalsahilai/portfolio-ai-voice-agent`

---

## Features

- Real-time microphone audio from the browser
- Live user transcription while the user is still speaking
- Deepgram streaming speech-to-text with interim results
- Retrieval-Augmented Generation (RAG) using Pinecone
- Grounded portfolio answers from indexed knowledge documents
- Multi-turn MongoDB conversation memory
- Three-phase memory/context strategy to control prompt size
- Gemini streaming response generation
- ElevenLabs WebSocket text-to-speech streaming
- Word/character-aligned agent captions synchronized with playback
- Pre-connected TTS to reduce response latency
- Parallel RAG, memory, and TTS connection work
- API-key/account rotation and fallback handling
- Browser mic suppression while the agent is speaking
- Clean WebSocket/session cleanup when a call ends
- Automatic MongoDB TTL cleanup for expired sessions

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Backend | FastAPI + Uvicorn | HTTP server and real-time WebSocket endpoint |
| Frontend | HTML, CSS, JavaScript | Browser voice-call interface |
| Transport | WebSocket | Bidirectional control messages and audio streaming |
| Speech-to-Text | Deepgram Nova-3 | Streaming transcription and interim results |
| LLM | Google Gemini | Streaming response generation and summaries |
| Retrieval | Pinecone | Semantic search over portfolio knowledge |
| Embeddings | Pinecone `llama-text-embed-v2` | Query/document vector representation |
| Memory | MongoDB | Session state, recent messages, and summaries |
| Text-to-Speech | ElevenLabs | Streaming voice generation |
| Audio playback | MediaSource + MP3 | Incremental browser playback |
| Logging | Python application logger | Runtime visibility and latency diagnostics |

---

## High-Level Architecture

```text
┌───────────────────────────────────────────────────────────────────┐
│                         Browser Frontend                          │
│                                                                   │
│  Microphone                                                       │
│      │                                                            │
│      ▼                                                            │
│  Web Audio API                                                    │
│      │  48 kHz browser audio                                     │
│      ▼                                                            │
│  Resample → PCM16 / 16 kHz                                       │
│      │                                                            │
│      └──────────── WebSocket /ws/call ───────────────────────┐    │
│                                                              │    │
│  Live user text ◀────────────────────────────────────────────┤    │
│  Agent audio + aligned captions ◀────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌───────────────────────────────────────────────────────────────────┐
│                         FastAPI Backend                           │
│                                                                   │
│  Deepgram STT                                                     │
│      │                                                            │
│      ├── Interim transcript ───────────────► Browser live text    │
│      │                                                            │
│      └── Final utterance                                          │
│             │                                                     │
│             ▼                                                     │
│       Call State Machine                                          │
│       LISTENING → THINKING → SPEAKING                             │
│             │                                                     │
│             ├────────► MongoDB memory                             │
│             ├────────► Pinecone RAG                               │
│             └────────► ElevenLabs pre-connect                     │
│                    (parallel work)                                │
│             │                                                     │
│             ▼                                                     │
│       Gemini context                                              │
│             │                                                     │
│             ▼                                                     │
│       Gemini streaming text                                      │
│             │                                                     │
│             ▼                                                     │
│       ElevenLabs streaming TTS                                   │
│             │                                                     │
│             ├── MP3 chunks ───────────────► Browser playback      │
│             └── alignment ────────────────► Live agent captions   │
└───────────────────────────────────────────────────────────────────┘
```

---

## Complete Conversation Flow

### 1. Call starts

The browser opens:

```text
ws://localhost:8001/ws/call
```

The backend:

1. accepts the WebSocket;
2. creates a call session;
3. creates the MongoDB session;
4. connects a Deepgram streaming STT session;
5. sends the greeting event to the frontend.

The greeting audio is played first. While the greeting is playing, microphone transmission is disabled so the assistant does not transcribe its own voice.

---

### 2. Browser captures the microphone

The browser uses `getUserMedia()` and the Web Audio API.

Typical browser microphone audio is 48 kHz. The frontend converts it to the format expected by the backend/STT pipeline:

```text
Browser microphone
     ↓
Float32 samples
     ↓
Resample to 16,000 Hz
     ↓
Convert to signed PCM16
     ↓
Send binary frames through WebSocket
```

Target format:

```text
Sample rate: 16000 Hz
Encoding: linear16 / PCM16
Channels: 1
```

---

### 3. Deepgram transcribes in real time

Deepgram runs as a persistent WebSocket connection for the call.

Important options include:

```text
model = nova-3
language = en-US
interim_results = true
smart_format = true
punctuate = true
vad_events = true
```

Because `interim_results=true`, the browser can show speech before the user finishes the sentence.

Example:

```text
You: Hello
You: Hello I want
You: Hello I want to know more
You: Hello I want to know more about Vishal
You: Hello I want to know more about Vishal and his experience.
```

The frontend updates the **same `You:` line** instead of adding one new line for every partial result.

When Deepgram marks the utterance complete, the final sentence is pushed to the AI processing pipeline.

---

## Call State Machine

The voice pipeline uses three main states:

```text
LISTENING
   │
   │ final user utterance
   ▼
THINKING
   │
   │ response generation begins
   ▼
SPEAKING
   │
   │ response finishes
   ▼
LISTENING
```

### No user interruption during agent speech

The current design intentionally prevents barge-in.

While the assistant is speaking:

- the frontend sets `canSendMic = false`;
- microphone frames are not sent;
- the backend also accepts microphone audio only while the call state is `LISTENING`.

The microphone resumes only after the agent audio has finished playing in the browser.

---

## Retrieval-Augmented Generation (RAG)

The agent does not depend only on the LLM's general knowledge. Portfolio information is stored in Pinecone and retrieved for each relevant final utterance.

Current runtime flow:

```text
User question
    ↓
Pinecone embedding request
    ↓
llama-text-embed-v2
    ↓
Pinecone index: voice-agent
    ↓
Similarity search
    ↓
top_k = 3
    ↓
Relevant chunks
    ↓
Gemini context
```

This allows the agent to answer questions such as:

- What experience does Vishal have?
- Which skills does he have?
- What projects has he built?
- What technologies does he use?
- How can I contact him?
- Where is his portfolio?

### Important embedding rule

The embedding model used when **ingesting documents** must match the embedding model used for **queries**.

The current runtime uses:

```text
llama-text-embed-v2
```

Do not mix vectors created by a different embedding model in the same retrieval space, even if the vector dimensions happen to match. If the embedding model changes, re-ingest the knowledge base.

---

## Updating Portfolio Knowledge in Pinecone

When your resume, experience, skills, projects, contact details, or portfolio content changes, update the source knowledge document and then re-run the ingestion pipeline.

A recommended project layout is:

```text
data/
└── portfolio.pdf          # or your current portfolio/resume knowledge file

scripts/
└── ingest.py
```

### Standard update workflow

1. Replace or edit the source document in the project's knowledge/data folder.
2. Activate the project virtual environment.
3. Run the ingestion script.
4. Confirm the vectors were written to the `voice-agent` Pinecone index.
5. Restart the backend if required by your local workflow.
6. Ask the agent a question that depends on the newly added information.

Run:

```bash
source venv/bin/activate
python scripts/ingest.py
```

Windows:

```powershell
venv\Scripts\activate
python scripts\ingest.py
```

### When replacing an existing document

The ingestion process should use deterministic vector IDs or clear the relevant old document vectors before inserting the new version. Otherwise old and new chunks can exist together and retrieval may return outdated information.

The safe logical process is:

```text
Updated document
      ↓
Remove/replace old vectors for that document
      ↓
Extract text
      ↓
Split into chunks
      ↓
Embed with llama-text-embed-v2
      ↓
Upsert into Pinecone index: voice-agent
      ↓
Verify retrieval
```

### Verify after ingestion

Start the application and ask a question that can only be answered by the updated document.

Example:

```text
User: What is Vishal's newest project?
```

Check the backend logs for messages similar to:

```text
Generating embeddings with model 'llama-text-embed-v2'
Querying index with top_k=3
RAG: retrieved 3 chunks
```

If retrieval still returns old facts, check that stale vectors were removed or overwritten during re-ingestion.

---

## Conversation Memory

MongoDB stores session-level memory.

Sessions use a TTL strategy and expire automatically after approximately two hours of inactivity.

The conversation context is intentionally compressed instead of continually sending the entire conversation to Gemini.

### Phase 1 — first user turn

```text
Current user message only
```

No previous conversation needs to be injected.

### Phase 2 — second user turn

```text
Previous user message
+ previous agent response
+ current user message
```

This preserves direct conversational continuity.

### Phase 3 — later turns

```text
Conversation summaries
+ current relevant context
+ current user message
```

Older exchanges are summarized and stored instead of repeatedly sending the entire raw history.

Benefits:

- smaller prompts;
- lower token usage;
- faster context construction;
- better long-conversation scalability;
- preserved conversational context.

Summary generation is performed as background work after the main voice response, so it does not need to block the current reply.

---

## Gemini Response Generation

The final user utterance, retrieved RAG context, and memory context are passed to Gemini.

Gemini uses streaming generation rather than waiting for the entire response.

```text
Gemini
  ↓
"Vishal "
  ↓
"is an AI "
  ↓
"Automation Engineer..."
```

Those text chunks are immediately forwarded to ElevenLabs instead of waiting for the complete answer.

This reduces perceived latency and allows TTS generation to begin earlier.

---

## ElevenLabs Streaming TTS

The system uses the ElevenLabs streaming WebSocket endpoint rather than generating one complete audio file first.

The connection enables synchronized alignment data:

```text
sync_alignment=true
```

The server receives packets containing:

```text
audio
alignment
is_final
```

The audio field is decoded and streamed to the browser immediately.

The alignment object contains character timing data such as:

```text
chars
char_start_times_ms
char_durations_ms
```

The frontend uses these timings together with the audio element's `currentTime` to reveal the agent's response as it is actually spoken.

Example:

```text
Audio says: "Vishal"
Screen:     Agent: Vishal

Audio says: "is an AI"
Screen:     Agent: Vishal is an AI

Audio says: "Automation Engineer"
Screen:     Agent: Vishal is an AI Automation Engineer
```

This is different from displaying the entire LLM response before the audio starts.

---

## Response-Time Optimization

Reducing perceived latency was a major design goal.

### Earlier sequential pipeline

The initial flow effectively behaved like this:

```text
Final transcript
      ↓
MongoDB + Pinecone RAG
      ↓
Build Gemini context
      ↓
Connect ElevenLabs WebSocket
      ↓
Start Gemini
      ↓
Generate TTS
      ↓
First audio
```

Each network step added to the previous one.

### Optimized parallel pipeline

The current implementation starts independent work at the same time:

```text
                        ┌── MongoDB message update
Final transcript ───────┼── Pinecone RAG
                        └── ElevenLabs WebSocket pre-connect
                                  │
                                  ▼
                           Build Gemini context
                                  │
                                  ▼
                           Gemini streaming
                                  │
                                  ▼
                      Already-connected ElevenLabs
                                  │
                                  ▼
                              First audio
```

The important optimization is that ElevenLabs connection time is hidden behind work that was already necessary for RAG and memory.

### Measured behavior during development

Observed local tests showed approximately:

```text
Memory + RAG:       ~1.5–2.3 seconds
Warm first audio:   ~3.3–4.4 seconds
Cold/warm-up turn:  can be ~5+ seconds
```

Actual latency varies with API/network conditions, model warm-up, Pinecone rate limits, Gemini first-token time, and browser playback startup.

### Additional latency optimizations already used

- persistent Deepgram WebSocket per call;
- Gemini streaming instead of full-response generation;
- ElevenLabs streaming instead of full-file synthesis;
- TTS WebSocket pre-connect in parallel with RAG;
- MongoDB work and Pinecone retrieval executed concurrently;
- background post-turn summarization;
- direct binary audio frames over the existing WebSocket;
- immediate MediaSource playback;
- live captions driven by audio alignment rather than waiting for final text.

---

## Live User Captions

Deepgram emits interim transcription events while the user speaks.

The backend forwards them to the browser as control messages such as:

```json
{
  "type": "user_speech_start"
}
```

```json
{
  "type": "user_transcript",
  "text": "Hello I want to know more",
  "is_final": false,
  "speech_final": false
}
```

At the end of the utterance:

```json
{
  "type": "user_transcript_end",
  "text": "Hello I want to know more about Vishal."
}
```

The frontend keeps one active user transcript element and updates its contents as the sentence grows.

---

## WebSocket Protocol

Main endpoint:

```text
/ws/call
```

### Browser → server

Binary WebSocket frames:

```text
PCM16 microphone audio at 16 kHz
```

### Server → browser control messages

Examples:

```text
session_started
play_greeting
user_speech_start
user_transcript
user_transcript_end
audio_start
audio_alignment
audio_end
stop_audio
error
```

### Server → browser binary frames

Binary frames contain ElevenLabs MP3 audio chunks.

---

## API/Account Rotation

The project is designed to work with multiple configured API credentials where supported.

### Deepgram

A new call can start from a different configured key slot. If a key fails, the session can move to another configured key.

### Gemini

Gemini request logic supports key rotation/fallback when a configured key is unavailable or exhausted.

### ElevenLabs

ElevenLabs account configuration can rotate between account entries containing an API key and voice ID.

Do not commit any API credentials to GitHub.

---

## Project Structure

The exact repository may evolve, but the core architecture is organized around these components:

```text
portfolio-ai-voice-agent/
│
├── main.py
│
├── api/
│   └── websocket_routes.py
│
├── audio/
│   └── stt_deepgram.py
│
├── call/
│   ├── call_state_machine.py
│   └── session_manager.py
│
├── llm/
│   └── gemini_service.py
│
├── memory/
│   ├── memory_manager.py
│   ├── summarizer.py
│   └── context_builder.py
│
├── rag/
│   ├── retriever.py
│   └── embeddings.py
│
├── tts/
│   └── voice_manager.py
│
├── config/
│   └── settings.py
│
├── utils/
│   └── logger.py
│
├── scripts/
│   └── ingest.py
│
├── data/
│   └── ... knowledge documents ...
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── requirements.txt
├── .env
└── README.md
```

---

## Local Setup

### Prerequisites

- Python 3.11 recommended
- Git
- Deepgram account/API key
- Gemini API key(s)
- ElevenLabs account/API key + voice ID
- Pinecone API key and `voice-agent` index
- MongoDB connection string
- Modern browser with microphone support

---

### 1. Clone the repository

```bash
git clone https://github.com/vishalsahilai/portfolio-ai-voice-agent.git
cd portfolio-ai-voice-agent
```

---

### 2. Create a virtual environment

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Core dependencies include FastAPI/Uvicorn, Deepgram-compatible WebSocket handling, Google GenAI, ElevenLabs, Pinecone, MongoDB/Motor, and environment/configuration packages.

---

### 4. Configure environment variables

Create `.env` from the repository example if one exists:

```bash
cp .env.example .env
```

Configure the values expected by `config/settings.py`.

Typical categories are:

```env
# Application
PORT=8001

# Deepgram
DEEPGRAM_API_KEY1=
DEEPGRAM_API_KEY2=
DEEPGRAM_API_KEY3=
DEEPGRAM_API_KEY4=
DEEPGRAM_MODEL=nova-3
DEEPGRAM_LANGUAGE=en-US
DEEPGRAM_ENDPOINTING_MS=350
DEEPGRAM_UTTERANCE_END_MS=1000

# Gemini
# Add the Gemini key variables expected by config/settings.py
GEMINI_MODEL=gemini-3.1-flash-lite

# ElevenLabs
# Add account API key(s) and voice ID(s) expected by config/settings.py
ELEVENLABS_MODEL_ID=eleven_flash_v2_5

# Pinecone
PINECONE_API_KEY=
PINECONE_INDEX_NAME=voice-agent

# MongoDB
MONGODB_URI=
```

> Never commit `.env` or real credentials.

---

### 5. Ingest the knowledge base

If the Pinecone knowledge base is not populated yet, or if the source document changed:

```bash
python scripts/ingest.py
```

Make sure ingestion and querying both use the same embedding model (`llama-text-embed-v2` in the current runtime).

---

### 6. Start the application

```bash
python main.py
```

Expected startup output includes messages similar to:

```text
Uvicorn running on http://0.0.0.0:8001
Pinecone ready ✅
MongoDB memory ready ✅
AI Voice Agent ready ✅
```

---

### 7. Open the browser correctly

Use:

```text
http://localhost:8001
```

Do **not** open:

```text
http://0.0.0.0:8001
```

`0.0.0.0` is the server bind address. `localhost` should be used in the browser so microphone APIs are available in the local secure-context exception supported by modern browsers.

Allow microphone permission when prompted.

---

## Development Test Checklist

Before deployment, verify:

- [ ] Page opens at `http://localhost:8001`
- [ ] Browser microphone permission works
- [ ] Greeting plays
- [ ] Mic remains blocked during greeting
- [ ] Live user words appear while speaking
- [ ] Final transcript is correct
- [ ] Pinecone retrieves relevant context
- [ ] Gemini response starts streaming
- [ ] ElevenLabs audio begins without waiting for the full answer
- [ ] Agent text appears in sync with spoken audio
- [ ] Mic remains blocked while agent speaks
- [ ] Mic resumes after browser playback ends
- [ ] Multi-turn memory works
- [ ] End Call cleans up the WebSocket and Deepgram session
- [ ] No uncaught traceback appears after normal call termination

---

## Ending a Call

When the visitor presses **End Call**:

1. browser playback is stopped;
2. microphone processing is stopped;
3. the client WebSocket is closed;
4. the backend receives the disconnect;
5. active call tasks are cancelled/cleaned up;
6. Deepgram is closed;
7. the in-memory call session is ended;
8. resources are released.

Normal logs may include:

```text
Client disconnected
connection closed
Deepgram session closed
Session ended
Session cleaned up
```

These messages are expected cleanup, not errors.

Background persistence/summarization that already started for a completed turn may finish independently.

---

## Troubleshooting

### Microphone access is not supported

Open:

```text
http://localhost:8001
```

instead of:

```text
http://0.0.0.0:8001
```

Then allow microphone access in the browser.

---

### Live user transcript appears as `Unknown message`

The browser is probably running an older cached `app.js`.

Hard refresh:

```text
macOS: Cmd + Shift + R
Windows/Linux: Ctrl + Shift + R
```

You can also open:

```text
http://localhost:8001/app.js
```

and verify that it contains handlers for:

```text
user_speech_start
user_transcript
user_transcript_end
```

---

### Agent answers but RAG quality is poor

Check:

1. the document was re-ingested after its latest update;
2. the index name is `voice-agent`;
3. ingestion and query use the same embedding model;
4. old document vectors were replaced/removed;
5. Pinecone returns relevant top-3 chunks.

A matching vector dimension alone does **not** mean two different embedding models are compatible.

---

### Response feels slow

Use the latency logs:

```text
Memory + RAG ready in X.XXs
FIRST AUDIO in X.XXs ✅
```

Interpretation:

```text
Large RAG time       → embedding/Pinecone/network bottleneck
Small RAG time but
large FIRST AUDIO    → Gemini first-token/TTS/network bottleneck
First request only   → cold-start/warm-up behavior
```

---

### Deepgram authentication failure

Check configured Deepgram keys. Remove or replace invalid key slots instead of relying permanently on fallback rotation.

---

### ElevenLabs stream failure

Check:

- API key/account quota;
- voice ID;
- model ID;
- WebSocket connectivity;
- `sync_alignment=true`;
- account rotation configuration.

---

## Deployment Notes

For production deployment:

- serve the frontend through HTTPS;
- use `wss://` for the voice WebSocket;
- use the hosting platform's assigned `PORT`;
- never expose API keys to frontend JavaScript;
- keep all Deepgram, Gemini, ElevenLabs, Pinecone, and MongoDB credentials on the server;
- configure allowed origins if frontend and backend are hosted separately;
- test browser microphone permissions on the final HTTPS domain.

Example production WebSocket concept:

```text
wss://your-backend-domain.com/ws/call
```

Do not hard-code `ws://localhost:8001/ws/call` for the deployed site.

---

## Security Notes

- Never commit `.env`.
- Revoke any API key accidentally exposed in Git history.
- Keep provider keys server-side only.
- Do not log complete secrets.
- Use separate development and production credentials where possible.
- Validate all client control messages.
- Keep dependency versions pinned for reproducible deployments.

---

## Performance Design Summary

The project is optimized around **time to first audible response**, not only total processing time.

The most important performance decisions are:

1. persistent Deepgram connection during a call;
2. interim STT for immediate user feedback;
3. parallel MongoDB + Pinecone + ElevenLabs connection work;
4. streaming Gemini generation;
5. streaming text directly into ElevenLabs;
6. streaming MP3 chunks directly to browser playback;
7. synchronized captions using ElevenLabs alignment;
8. background memory summarization after the critical response path.

The resulting pipeline behaves more like a real voice assistant than a traditional request/response chatbot.

---

## Future Improvements

Potential future improvements include:

- production deployment on Render or another persistent backend;
- dynamic production WebSocket URL selection;
- optional phone-number integration through Twilio/Telnyx/SIP;
- RAG skip rules for greetings, thanks, and goodbye messages;
- response caching for repeated portfolio questions;
- persistent/pre-warmed TTS connection strategies where provider behavior allows it;
- retrieval evaluation and automated RAG regression tests;
- admin endpoint/tool for safe document re-ingestion;
- automated knowledge-base sync when resume/portfolio files change;
- call analytics and latency dashboards.

---

## Author

**Vishal Sahil**  
AI Automation Engineer · AI Agent Developer · Prompt Engineer

- Portfolio: `https://vishalsahilai.vercel.app`
- GitHub: `https://github.com/vishalsahilai`

---

## License

Add the repository's chosen license here. If the project is intended to be open source, include a `LICENSE` file in the repository and reference it from this section.

---

## Quick Command Reference

```bash
# Clone
git clone https://github.com/vishalsahilai/portfolio-ai-voice-agent.git
cd portfolio-ai-voice-agent

# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Install
pip install -r requirements.txt

# Configure
cp .env.example .env

# Rebuild/update Pinecone knowledge
python scripts/ingest.py

# Run
python main.py

# Open
# http://localhost:8001
```

---

**Sada** is designed as a portfolio-native real-time AI voice assistant: visitors can speak naturally, watch both sides of the conversation appear live, and receive grounded answers backed by a RAG knowledge base and multi-turn memory.
