"""
Prompt templates for LoCoGen pipeline.

This module contains all prompt templates used in the 5-stage
data generation pipeline, extracted from the original code and
organized for easy maintenance and modification.
"""

from typing import Dict, Any


class PromptTemplates:
    """Collection of all prompt templates used in LoCoGen pipeline."""

    # ========================================================================
    # STAGE 1: Character Initialization
    # ========================================================================

    CHARACTER_INIT_TEMPLATE = """Please create fictional character situations at three different time points (1 year ago, 3 years ago, 5 years ago) based on the character information provided below.

Use brief sentences to describe each time point's character situation.

Each time point must contain unique information and should reflect the alternating development of new and old things (e.g., new hobbies, further development of old interests, formation of new relationships, personality changes, etc.).

The information should be appropriate for the character's age at that time. Please describe information ("hobby", "personality", "family_relationship", "social_relationship", "study_or_work_status") in a concise paragraph:

{character_info}

Generate a standard json format:
{{
  "YYYY-MM-DD": {{
    "name": "...",
    "gender": "...",
    "age": "...",
    "hobby": "...",
    "personality": "...",
    "family_relationship": "...",
    "social_relationship": "...",
    "study_or_work_status": "..."
  }},
  "YYYY-MM-DD": {{
    "name": "...",
    "gender": "...",
    "age": "...",
    "hobby": "...",
    "personality": "...",
    "family_relationship": "...",
    "social_relationship": "...",
    "study_or_work_status": "..."
  }},
  "YYYY-MM-DD": {{
    "name": "...",
    "gender": "...",
    "age": "...",
    "hobby": "...",
    "personality": "...",
    "family_relationship": "...",
    "social_relationship": "...",
    "study_or_work_status": "..."
  }}
}}"""

    CHARACTER_DETAIL_EXPANSION_TEMPLATE = """Below are two character profiles from different points in time.
Please insert {{N}} additional profiles at different points in time between the given profiles, showcasing the progression and alternation of new and old elements (such as developing new hobbies, furthering existing interests, forming new relationships, personality changes, etc.).

The profiles must fit the character's age at that time, demonstrating their development and changes to make the transitions more natural and complete. Only reply with {{N}} character profiles.

{{time1_info}}
{{time2_info}}"""

    # ========================================================================
    # STAGE 2: Diary Generation
    # ========================================================================

    DIARY_GENERATION_TEMPLATE = """Please generate {{n}} coherent diary entries for the character based on the following information, with each entry occurring between the specified two time points.

Each diary entry should include a date and content, and refer to the context provided to ensure coherence and consistency.

{{
[Part 1: Background Information]
{{structured_data_list}}
[Part 2: Descriptions of specified two time points]
time1 describe: {{time1_describe}}
time2 describe: {{time2_describe}}
[Part 3: Summaries of previous diary entries]
{{diaries_summary}}
[Part 4: Recent Diary Content]
{{last_stage_diaries}}
}}

When generating new diary entries, please follow these requirements:
{{
1. Each diary entry's time point should be evenly distributed between [time1 describe] and [time2 describe].
2. The diary content should reflect the character's changes and development from time point 1 to time point 2.
3. The diary content should not conflict with the Background Information, Summaries of previous diary entries, and Recent Diary Content.
4. Each diary entry must describe a specific event, and any mentioned locations, people, or items must have specific names.
}}"""

    DIARY_SUMMARY_TEMPLATE = """Please read the following diary contents and summarize all the key information from the diaries. Remove any invalid or redundant expressions, retaining only the core content of each diary. The diary contents are as follows:
{{
{{events_content}}
}}
Please output a paragraph summarizing what is discussed in all the diaries. Must be less than 500 words."""

    STRUCTURED_DATA_UPDATE_TEMPLATE = """Author's past situation:
{{past_elements}}
Author's recent diary:
{{
{{events_content}}
}}
Please update the [author's past situation] based on the [author's recent diary], ensuring the content is updated to the latest descriptions for each item. For content that has changed (educational background, emotional status), keep only the most recent one.

Please output in JSON format, including [social circle list, family relationship list, study or work progress, educational background, emotional status]."""

    # ========================================================================
    # STAGE 3: Dialogue Generation
    # ========================================================================

    DIALOGUE_GENERATION_TEMPLATE = """Please construct a multi-turn dialogue (3-5 rounds) record between a user and a chatbot based on the following the user's diary entry, with the conversation occurring at the same time as described in the diary:
{{the_event}}

Requirements:
1. The Chatbot's responses should be conversational, logically clear, and varied.
2. The format must refer to: {{formatted_data}}
3. The chat must be coherent, brief and natural."""

    # ========================================================================
    # STAGE 4: Dataset Construction
    # ========================================================================

    # (Dataset construction typically doesn't use prompts, it's data processing)

    # ========================================================================
    # STAGE 5: Question Generation
    # ========================================================================

    QA_GENERATION_TEMPLATE = """The current time is {{current_time}}.
The following is a historical conversation between the user and the chatbot: {{history_conversation}}

Task: Please choose a key piece of information from the historical conversation (e.g., the name of an event, a person's name, a location, etc.), and then construct a question and answer pair between the user and the chatbot based on that key information.

In the question, the user needs to provide a detailed and specific description to ensure the answer is clear and precise, guiding the chatbot to provide an accurate response based on the historical conversation.

The chatbot must use the key information mentioned in the historical conversation as part of its reply.

Please output a structured JSON object following this format: {{"User": "A detailed, accurate question.", "Chatbot": "Response."}}"""

    # ========================================================================
    # EVALUATION & QUALITY CONTROL
    # ========================================================================

    DIALOGUE_QUALITY_CHECK_TEMPLATE = """Check whether the conversation data meets the following conditions. If yes, output Yes; otherwise, output No:

1. Incomplete conversations: Any missing or incomplete conversation records should be filtered out.
2. Noisy conversations: Any conversations that contain obvious noise, such as typos, grammatical errors, or non-linguistic characters, should be filtered out to improve data quality and model training efficiency.

{{conversation_data}}"""

    DIALOGUE_EVALUATION_TEMPLATE = """Context:
You are an evaluator tasked with assessing the quality of a conversation between a user and a chatbot. You need to rate the conversation based on three metrics: Participation, Coherence, and Rationality.

Instructions:
Participation: Rate how actively and meaningfully both parties (user and chatbot) engage in the conversation. Consider the relevance and contribution of each turn in the dialogue.

Coherence: Evaluate the logical flow and consistency of the conversation. The dialogue should make sense as a whole, with each response appropriately following the preceding interaction.

Rationality: Assess the reasonableness and sensibility of the chatbot's responses. The responses should be logical, well-founded, and appropriate given the context of the conversation.

For each metric, provide a score on a scale from 1 to 5, where 1 is very poor and 5 is excellent.

Example Conversation: {{conversation}}

Evaluation Format:
{{
  "Participation": [Your Score],
  "Coherence": [Your Score],
  "Rationality": [Your Score]
}}"""

    CONSISTENCY_CHECK_TEMPLATE = """Record of the conversation between the user and the chatbot: {{history_conversation}}
The current time is: {{current_time}}

Now, the user asks the chatbot a question to check if the chatbot remembers something mentioned in the record of the conversation: {{question}}

The response of the chatbot is: {{response}}

Please determine whether the response of the chatbot is accurate. If the response of the chatbot is consistent with the content in the record of the conversation, please output "Yes", otherwise output "No"."""

    INFORMATION_CLASSIFICATION_TEMPLATE = """Please categorize the answers to the questions. Categories need to be selected from ["people", "date and time", "location", "event", "emotions", "entity"]. You only need to output the category of the answer information.
{{
{{question}}
Answer: {{answer}}
}}
Class:"""

    # ========================================================================
    # UTILITY PROMPTS
    # ========================================================================

    JSON_FORMAT_FIX_TEMPLATE = """The following content is not in a standard json format. Please format it in a standard json format. You only need to reply with a json format content:
{content}"""

    @classmethod
    def format_character_init_prompt(cls, character_info: str) -> str:
        """
        Format character initialization prompt.

        Args:
            character_info: Character information string

        Returns:
            Formatted prompt
        """
        return cls.CHARACTER_INIT_TEMPLATE.format(character_info=character_info)

    @classmethod
    def format_character_expansion_prompt(
        cls,
        n: int,
        time1_info: str,
        time2_info: str
    ) -> str:
        """
        Format character detail expansion prompt.

        Args:
            n: Number of profiles to generate
            time1_info: Time point 1 information
            time2_info: Time point 2 information

        Returns:
            Formatted prompt
        """
        return cls.CHARACTER_DETAIL_EXPANSION_TEMPLATE.format(
            N=n,
            time1_info=time1_info,
            time2_info=time2_info
        )

    @classmethod
    def format_diary_generation_prompt(
        cls,
        n: int,
        structured_data_list: str,
        time1_describe: str,
        time2_describe: str,
        diaries_summary: str,
        last_stage_diaries: str
    ) -> str:
        """
        Format diary generation prompt.

        Args:
            n: Number of diary entries to generate
            structured_data_list: Structured data list
            time1_describe: Time point 1 description
            time2_describe: Time point 2 description
            diaries_summary: Summary of previous diaries
            last_stage_diaries: Recent diary content

        Returns:
            Formatted prompt
        """
        return cls.DIARY_GENERATION_TEMPLATE.format(
            n=n,
            structured_data_list=structured_data_list,
            time1_describe=time1_describe,
            time2_describe=time2_describe,
            diaries_summary=diaries_summary,
            last_stage_diaries=last_stage_diaries
        )

    @classmethod
    def format_dialogue_generation_prompt(
        cls,
        the_event: str,
        formatted_data: str
    ) -> str:
        """
        Format dialogue generation prompt.

        Args:
            the_event: Event description from diary
            formatted_data: Formatted dialogue data example

        Returns:
            Formatted prompt
        """
        return cls.DIALOGUE_GENERATION_TEMPLATE.format(
            the_event=the_event,
            formatted_data=formatted_data
        )

    @classmethod
    def format_qa_generation_prompt(
        cls,
        current_time: str,
        history_conversation: str
    ) -> str:
        """
        Format QA generation prompt.

        Args:
            current_time: Current timestamp
            history_conversation: Historical conversation

        Returns:
            Formatted prompt
        """
        return cls.QA_GENERATION_TEMPLATE.format(
            current_time=current_time,
            history_conversation=history_conversation
        )

    @classmethod
    def format_consistency_check_prompt(
        cls,
        history_conversation: str,
        current_time: str,
        question: str,
        response: str
    ) -> str:
        """
        Format consistency check prompt.

        Args:
            history_conversation: Historical conversation
            current_time: Current time
            question: Question asked
            response: Chatbot response

        Returns:
            Formatted prompt
        """
        return cls.CONSISTENCY_CHECK_TEMPLATE.format(
            history_conversation=history_conversation,
            current_time=current_time,
            question=question,
            response=response
        )


# Convenience aliases
Prompts = PromptTemplates
