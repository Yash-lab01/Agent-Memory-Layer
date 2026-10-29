# Spec 019: WanderAI Travel Companion Integration Demo

## 1. Goal
Provide a production-grade vertical integration of **Agent Memory Layer (`memlayer`)** into **WanderAI**, an AI-powered conversational travel planning assistant. This demo showcases how durable memory transforms a stateless chatbot into an intelligent travel companion that:
1. **Remembers traveler preferences across different trips & sessions** (dietary needs, airline alliances, hotel vibes, budget ranges, pace of travel).
2. **Handles preference updates & contradictions non-destructively** (e.g., switching airline loyalty programs or accommodation styles while preserving historical provenance).
3. **Performs multi-hop destination reasoning via the Knowledge Graph Layer** (e.g., connecting traveler tastes to specific neighborhoods, attractions, and culinary spots).
4. **Enforces strict tenant isolation and user privacy** (`user_id` scoping, explicit forgetting, and exportable traveler profile).

---

## 2. Architecture & Design

### 2.1 WanderAI Travel Agent (`demo/wanderai/agent.py`)
The `WanderAIAgent` wraps `MemoryLayer` with travel-specific capabilities:
- **Traveler Profile Synthesis**: Extracts and organizes traveler facts into structured preference categories (`dietary`, `budget`, `loyalty`, `interests`, `accommodation`, `past_trips`).
- **Context Injection**: Enhances travel planning prompts with relevant active traveler memories and destination graph relationships.
- **Contradiction Resolution**: Detects updates to travel preferences (e.g., "I'm now strictly gluten-free" superseding "I love Italian pasta joints") using `ConflictDetector`.

### 2.2 Pre-Configured Showcase Scenarios (`demo/wanderai/scenarios.py`)
To make recruiter and developer evaluation effortless, the demo includes a 4-turn canonical cross-trip scenario:
1. **Turn 1 (Tokyo Spring Trip - Setup)**:
   - *"I'm planning a 7-day trip to Tokyo. I am vegan, have Star Alliance Gold status, and love contemporary art museums."*
   - Facts extracted and stored with entity tags `dietary`, `airline`, `interests`.
2. **Turn 2 (Preference Update / Contradiction)**:
   - *"Actually, I switched my airline loyalty to OneWorld Sapphire, and my hotel budget is strictly under $150/night."*
   - Star Alliance Gold memory is marked `superseded` non-destructively.
3. **Turn 3 (Paris Fall Trip - 6 Months Later)**:
   - *"Help me plan a 4-day autumn trip to Paris. Recommend a hotel and a dinner spot."*
   - MemLayer injects active vegan dietary preferences, OneWorld status, and <$150 hotel budget. Zero amnesia; agent personalizes recommendations automatically.
4. **Turn 4 (Multi-Hop Knowledge Graph Recommendation)**:
   - Graph relationships connect Paris neighborhoods (e.g. *Le Marais* -> *KNOWN_FOR* -> *Vegan Bakeries*, *Le Marais* -> *NEARBY* -> *Centre Pompidou Modern Art*).
   - Agent synthesizes an itinerary tailored to art and plant-based dining.

### 2.3 Standalone CLI Demo (`demo/wanderai/run_demo.py`)
A single-command script `python -m demo.wanderai.run_demo` that executes the 4-turn dialogue offline (or with live LLM when configured), displaying colored before-and-after memory states, prompt token savings, and supersession links.

### 2.4 Web Demo Endpoints (`demo/app/main.py`)
- `POST /api/wanderai/chat`: Travel dialogue endpoint supporting memory toggle and traveler ID.
- `GET /api/wanderai/profile`: Returns structured traveler preferences (active facts, superseded history, visited destinations).
- `GET /api/wanderai/scenarios`: Returns predefined demo scenarios for interactive UI one-click playback.

---

## 3. Acceptance Criteria
1. `WanderAIAgent` successfully wraps `MemoryLayer` and supports asynchronous extraction and multi-hop graph querying.
2. Contradictory travel preferences (e.g., airline loyalty, budget changes) are properly superseded without data loss.
3. Cross-trip retrieval accurately retrieves relevant preferences across different destination prompts without prompt leakage.
4. `python -m demo.wanderai.run_demo` runs to completion and exits with code 0.
5. All new endpoints in `demo/app/main.py` are covered by automated tests in `tests/test_wanderai.py`.
6. Strict type checking (`mypy`), linting (`ruff`), and documentation builds (`mkdocs`) pass with 0 errors.
