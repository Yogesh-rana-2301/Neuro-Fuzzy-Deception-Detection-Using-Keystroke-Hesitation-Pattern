"""
src/collection/prompts.py
-------------------------
Standardized question prompts for collecting truthful vs. fabricated answers.
Used in data collection protocols and live demonstrations.
"""

TRUTHFUL_PROMPTS = [
    {
        "id": "T01",
        "category": "Daily Routine",
        "prompt": "What did you eat for breakfast today? Describe it in 1-2 sentences."
    },
    {
        "id": "T02",
        "category": "Past Experience",
        "prompt": "Name the city you grew up in and mention one memorable landmark there."
    },
    {
        "id": "T03",
        "category": "Personal Preference",
        "prompt": "What is your favorite movie or book, and why do you genuinely enjoy it?"
    },
    {
        "id": "T04",
        "category": "Recent Memory",
        "prompt": "Describe what you were doing yesterday at around 8:00 PM."
    },
    {
        "id": "T05",
        "category": "Education / Work",
        "prompt": "What was the very first course or topic you studied this semester?"
    }
]

FABRICATION_PROMPTS = [
    {
        "id": "F01",
        "category": "Invented Meal",
        "prompt": "Invent a breakfast you did NOT eat today, but make it sound completely convincing."
    },
    {
        "id": "F02",
        "category": "Fictional Trip",
        "prompt": "Describe a vacation to a foreign country you have never visited as if you just returned."
    },
    {
        "id": "F03",
        "category": "False Alibi",
        "prompt": "Fabricate an alibi for yesterday at 10:00 PM: invent believable people and actions."
    },
    {
        "id": "F04",
        "category": "Made-up Pet",
        "prompt": "Invent an unusual pet you have never owned. Give it a name and describe a habit."
    },
    {
        "id": "F05",
        "category": "Fictitious Event",
        "prompt": "Describe an imaginary celebrity encounter you had in a coffee shop."
    }
]
