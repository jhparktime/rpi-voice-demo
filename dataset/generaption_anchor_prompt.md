I am building an intent classification router for an edge AI robot. I need a dataset of sentences categorized into two classes:

Class A: Local_ChitChat (For sLLM)
Casual conversation, emotional reaction, simple greetings.
Needs high empathy but low logic.
Examples: "I feel so sad today", "Wow, that's amazing!", "Hi, how are you?"
Class B: Cloud_Knowledge (For GPT-4)
Fact retrieval, logical reasoning, complex tasks.
Needs external knowledge or computation.
Examples: "Who is the president of USA?", "Explain quantum physics", "Translate this to Korean."
Request: Generate 50 distinct sentences for EACH class (Total 100).
Diversity: Use various sentence lengths and diverse topics.
Format: Provide the output as a clean JSON format with keys "local_anchors" and "cloud_anchors".
Do NOT number the lines, just give me the JSON array.