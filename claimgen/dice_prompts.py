# Adapted from claim-augmented-generation/data_prep/decontext/prompts.py (English version).

system_prompt = """
You are InformationExtractor, an intelligent assistant that extracts structured information from text by decomposing it into a list of standalone factual statements.
""".strip()

# Revised prompt: decontextualized, atomic claims; '<c>' tags; no statements about the passage itself.
user_prompt_claim = """
You will be given a passage from a web page. Your task is to extract the factual claims it makes.

Instructions:
- Enclose each extracted claim with the tags '<c>' and '</c>'.
- Each claim must be supported by the passage. Do not add information that is not in the passage.
- Each claim must be understandable on its own, without the passage. Replace pronouns (e.g., it, they, we, our, you) and vague references (e.g., this, these, the study, the company) with the specific entity they refer to. Use the title to identify the entity if needed.
- Each claim should state a single fact. Split sentences that contain several facts into separate claims.
- State the facts themselves, not what the passage does. Write "Sea levels are rising.", not "The article discusses rising sea levels."
- Do not write claims about the passage, article, section, or its author (e.g., what it discusses or covers); skip such sentences.
- Ensure the claims together cover all the key facts in the passage.
- Only generate claims, do not include any additional texts or explanations.

Title: {title}
Passage:
{text}
""".strip()

user_prompt_claim_no_title = user_prompt_claim.replace("Title: {title}\n", "")
