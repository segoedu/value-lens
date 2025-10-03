"""
Configuration file for LLM Value Analysis app
"""

# Available LLM models from Groq
AVAILABLE_MODELS_GROQ = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "qwen/qwen3-32b"
]

# Available LLM models from Openrouter
AVAILABLE_MODELS_OPENROUTER = [
    "deepseek/deepseek-chat-v3.1:free",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "google/gemma-3-27b-it",
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-5-nano",
    "openai/gpt-5-mini",
    "openai/gpt-5"
]

# Example texts for "Generate Random Text" feature
EXAMPLE_TEXTS = [
    "I've been working nights and weekends on this app idea because I want to finally build something that's completely mine. I don't care if it fails—I need to know I tried. I've always felt trapped following other people's rules, and this is my way of breaking free and proving I can do it.",
    "Water boils at 100 degrees Celsius at sea level. The chemical formula for water is H₂O. These properties change under varying pressure.",
    "I stopped eating meat and started composting because the climate crisis is real. If we don't all make some sacrifices now, future generations are going to pay the price.",
    "I chose investment banking not because I love finance, but because it pays well and gets me into high-status rooms. I want the kind of wealth that gives me options.",
    "When my sister got sick, I moved back home to take care of her. It wasn't even a question—I'll always be there for family, no matter what I have to give up.",
    "Every year, we celebrate the harvest festival with the same rituals my grandparents used. It's how we stay connected to our roots.",
    "I volunteer with an organization that helps asylum seekers. Everyone deserves safety and dignity, no matter where they're from.",
    "Last summer, I quit my job and traveled through South America. I just wanted to enjoy life, meet people, try new foods, and feel truly alive.",
    "I chose a quieter neighborhood because it feels safer, especially with kids. I need to know they're protected.",
    "In politics, perception is everything. You can't afford to look weak or lose control. I'm careful about how I present myself in public."
]

# Prompts for analysis
PROMPTS = {
    "value_filtering": """
    You will analyze the following TEXT to assess the presence and subjective centrality of a set of human values. The values are drawn from a structured value theory provided below. Your goal is to identify whether any values are expressed in the text and to assess how subjectively central those values are.

    <text>{text}</text>

    Now use this VALUE THEORY to interpret the text:
    <value_theory>{value_theory}</value_theory>

    Your task:
    Determine whether the text contains implicit or explicit references to any of the values listed in the value theory. A value may be expressed directly (e.g., "I value independence") or indirectly through actions, preferences, goals, or emotional tone (e.g., "I want to start my own company" may imply Self-Direction). You must evaluate **how central** any identified values are to the meaning and motivation in the text.

    Use the following 4-level framework to assess **value presence and subjective centrality**:

    <value-salience-and-subjectivity-scale>
    VALUE NEUTRAL / NON-SUBJECTIVE
    **Level 1 – PURELY OBJECTIVE (No value content)**
    - Text contains only verifiable facts or neutral descriptions
    - No identifiable values or value-laden judgments
    - No interpretation of preferences, goals, or motivations required

    **Level 2 – LOW VALUE SALIENCE (Incidental or background)**
    - Value references may be present but are incidental or not essential to the core message
    - The values are peripheral and could be removed without altering the intent or meaning
    - Examples: mentioning a personal interest without expressing preference; describing others' values but not endorsing them

    VALUE-CENTERED / SUBJECTIVE
    **Level 3 – MODERATE VALUE SALIENCE (Relevant and shaping content)**
    - One or more values are expressed and influence the content
    - The meaning of the text partially depends on these values
    - Multiple valid interpretations are possible based on value orientation
    - The text perspective or shown preferences are important to the message

    **Level 4 – HIGH VALUE SALIENCE (Core to meaning)**
    - The entire text is centered around expressing, justifying, or acting on a personal value or set of values
    - Values are the central organizing principle of the text
    - Text reflects personal convictions, identity, or strong preferences
    </value-salience-and-subjectivity-scale>

    Decision process:
    1. Carefully read the text and identify any values referenced in the text (explicitly or implicitly).
    2. Judge how central each identified value is to the meaning or motivation of the text.
    3. Classify each present value into Level 1, 2, 3, or 4 based on its **subjective salience**.
    4. Output a JSON array **only of the values with `"present": true"`**, with the following fields:

    ```json
    [
      {{
        "id": "<value id, e.g. SDI>",
        "name": "<value name, e.g. Self-Direction>",
        "present": true,
        "subjectivity_level": 1 | 2 | 3 | 4,
        "evidence": ["<quote1>", "<quote2>", …]
      }}
    ]
    ```

    If no values are present (i.e. all subjectivity levels are 0 or undefined), output an empty array `[]`.
    Do not include any other commentary or text outside the JSON.
    """,
    
    "value_measuring": """
    You are a value intensity evaluator. Your task is to assess how strongly a given human value is either supported or opposed in a piece of text, using a predefined scale.

    If the <detected_values> list is empty, return an empty list []. Only analyze values that are explicitly present in the <detected_values> list.

    You will receive three inputs:
    1. A value theory that defines core human values
    2. A text to analyze
    3. A list of detected values present in the text, with identifiers matching those from the value theory.

    ---

    VALUE THEORY:
    <value_theory>{value_theory}</value_theory>

    ---

    TEXT TO ANALYZE:
    <text>{text}</text>

    ---

    DETECTED VALUES:
    <detected_values>{detected_values}</detected_values>

    ---

    Use the following INTENSITY SCALE to evaluate how forcefully each detected value is expressed in the text:

    <intensity-scale>
    INTENSITY SCALE (Value force estimation)
    - strong_support: The text actively and passionately promotes, defends, emphasis or endorses the value. The value is central to the message, reinforced with emotional, moral, or logical urgency.
    - mild_support: The text shows clear but gentle alignment with the value through positive mention or subtle endorsement, without significant elaboration, insistence or emphasis.
    - neutral: The text mentions or touches on the value without signaling any clear support or opposition. The tone is factual, balanced, or incidental.
    - mild_resistance: The text subtly questions, downplays, or introduces alternative perspectives to the value. This opposition is indirect, hedged, or expressed through soft skepticism.
    - strong_resistance: The text directly and forcefully challenges, criticizes, or undermines the value. This includes explicit argumentation, negative emotional tone, or repeated rejection.
    - reframing: The text acknowledges the value but redirects its meaning, context, or significance. It introduces a new perspective that shifts emphasis without outright support or opposition.
    - no_values: The text is purely technical or descriptive, with no discernible value content or evaluative stance.
    </intensity-scale>

    For **each detected value**, evaluate the **degree of intensity** with which the text supports or opposes that value. Use only the scale provided above. Your evaluation should be grounded in textual evidence such as rhetorical emphasis, emotional tone, framing, repetition, placement, and any normative language that affirms or undermines the value.

    Return your output as a list of JSON objects, one per detected value, in the following format:

    ```json
    [
      {{
        "id": "<ID from value theory>",
        "name": "<Name of the value>",
        "intensity": "<one of: strong_support | mild_support | neutral | mild_resistance | strong_resistance | reframing | no_values>",
        "justification": "<Concise explanation grounded in textual evidence>"
      }}
    ]
    ```

    Be concise but precise in your justification. You are not judging whether the value is good or bad; your job is only to measure how strongly it is promoted or resisted in the text.
    """
}