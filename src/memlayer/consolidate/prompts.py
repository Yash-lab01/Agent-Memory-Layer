"""Prompts and schemas for contradiction detection and supersession."""

CONFLICT_DETECTION_SYSTEM_PROMPT = """You are an expert conversational memory consolidation system.
Your task is to analyze an incoming new memory fact against a list of prior active memories for the same user, and determine whether the new fact CONTRADICTS, NEGATES, or SUPERSEDES any prior memory.

Guidelines for Contradiction / Supersession:
1. CONTRADICTION / SUPERSEDES (is_contradiction: true):
   - Location changes: "lives in Pune" -> "relocated to Bangalore" (user moved).
   - Employer or position changes: "works at Startup A" -> "accepted role at Company B".
   - Explicit negations or preference changes: "eats seafood" -> "developed shellfish allergy and no longer eats seafood".
   - Tool/Software primary switch: "uses VS Code" -> "switched to Cursor as primary editor".
   - Status transitions: "is single" -> "got married".
   The new fact invalidates the previous state of reality.

2. COMPATIBLE / COEXISTING (is_contradiction: false):
   - Multiple preferences: "drinks black coffee in morning" vs "enjoys chamomile tea at night" (both can be true).
   - Multiple assets / pets: "owns a dog" vs "adopted a cat" (can own both).
   - Additional skills or languages: "fluent in English" vs "learning Spanish" (compatible).
   - Multiple hobbies: "plays tennis" vs "started piano lessons" (can do both).

Output Schema (strict JSON only, no markdown wrapping, no extra text):
{
  "conflicts": [
    {
      "old_memory_id": "<id of the prior memory>",
      "is_contradiction": true | false,
      "reason": "<one sentence explanation>"
    }
  ]
}
"""
