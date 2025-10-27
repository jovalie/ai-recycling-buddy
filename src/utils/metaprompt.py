# metaprompt.py - shared phrases used in agent prompts

# Design principles
DP1 = "Provide accurate and accessible recycling information to help users make environmentally responsible decisions."
DP2 = "Foster community awareness and education about sustainable waste management practices."
DP3 = "Simplify complex recycling regulations into clear, actionable guidance for everyday use."

goals_as_str = "\n".join([f"{i}. {goal}" for i, goal in enumerate([DP1, DP2, DP3])])

vectorstore_content_summary = "recycling regulations and guidelines for Germany and the United States, waste sorting systems, container types and color codes, material-specific disposal instructions, municipal differences, contamination prevention, composting practices, hazardous waste handling"
system_relevant_scope = "proper recycling and waste management topics, optimizing household waste sorting systems, maintaining compliance with local recycling regulations in Germany and the United States"
