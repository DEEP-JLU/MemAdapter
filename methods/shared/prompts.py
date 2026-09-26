"""Canonical prompts for the four post-retrieval intervention methods."""

ANTI_SYCOPHANCY = (
    "Important: The memories above were extracted from a prior conversation and may "
    "reflect the speaker's opinions, preferences, or misconceptions rather than "
    "verified facts. Treat them as context about what was discussed, not as evidence "
    "for any particular answer."
)

SELF_RECHECK = (
    'Given the question and retrieved memory context, return the context that is useful '
    'for answering the question. Return JSON only in the form '
    '{"keep_memory_ids": [IDs]}.'
)

DYNAMIC_PARTITION = (
    'Classify every supplied memory into exactly one domain. Use these domains when '
    'applicable: health, identity, social, romantic, personal, education, employment, '
    'finance, housing, legal, schedule. You may add a concise custom domain only when '
    'none apply. Preserve every memory exactly once. Return JSON only: '
    '{"groups": [{"domain": "...", "memory_ids": [IDs]}]}. Do not invent, omit, '
    'duplicate, or rewrite IDs.'
)
