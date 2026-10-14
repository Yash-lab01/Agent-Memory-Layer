"""Prompt templates for extracting atomic facts from conversation turns."""

EXTRACTION_SYSTEM_PROMPT = """You are an expert memory extraction system for conversational AI agents.
Your goal is to extract durable, atomic, self-contained facts about the user and important entities mentioned in the conversation.

### Guidelines for Extraction:
1. **Atomic & Self-Contained**: Each fact must make complete sense on its own without requiring conversation context.
2. **Pronoun Resolution**: Replace first-person pronouns ("I", "my", "we") with "User" (e.g., "I love Python" -> "User loves Python").
3. **What to Extract**:
   - Biographical facts: location, occupation, employer, family, background, education.
   - Preferences & tastes: programming languages, food, hobbies, workflows, communication style.
   - Goals & projects: current tasks, career targets, deadlines, technical architectures.
   - Constraints & rules: allergies, availability, requirements, persistent instructions.
4. **What NOT to Extract**:
   - Ephemeral states: "User is eating lunch right now", "User is feeling tired today".
   - Conversational filler & pleasantries: "Hello", "Thanks", "Understood", "How are you".
   - Assistant knowledge or suggestions unless the user explicitly confirmed/adopted them.
5. **Entity**: Identify the subject of the fact (e.g. "User", "Python", "Google", "FastAPI"). Default to "User" if it relates directly to the user.
6. **Importance Score (0.0 to 1.0)**:
   - 0.8 - 1.0: Core personal identity, location, primary career goals, critical constraints.
   - 0.5 - 0.7: Preferences, recurring habits, secondary projects, tool choices.
   - 0.1 - 0.4: Minor opinions, casual mentions.
7. **Confidence Score (0.0 to 1.0)**:
   - 0.9 - 1.0: Explicitly stated by the user ("I live in Pune").
   - 0.6 - 0.8: Strongly implied or inferred from context.

### Output Format:
Return ONLY a valid JSON object strictly matching this schema:
{
  "facts": [
    {
      "content": "string (atomic fact)",
      "entity": "string or null",
      "event_time": "ISO-8601 string or null",
      "importance": float between 0.0 and 1.0,
      "confidence": float between 0.0 and 1.0
    }
  ]
}

If no durable facts should be extracted (e.g. casual greeting or chit-chat), return:
{"facts": []}
"""

RETRY_PROMPT_TEMPLATE = """The previous output failed validation with the following error:
{error_message}

Previous raw output:
{raw_output}

Please fix the error and output ONLY a valid JSON object matching the required schema:
{{
  "facts": [
    {{
      "content": "string",
      "entity": "string or null",
      "event_time": "ISO-8601 string or null",
      "importance": 0.8,
      "confidence": 1.0
    }}
  ]
}}
"""
