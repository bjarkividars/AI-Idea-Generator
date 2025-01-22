import logging
import json
from typing import List, Optional
from my_openai import OpenAIClient

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class IdeaItem:
    """
    A Python class to store a brainstorming idea with title, description, 
    and an optional rating (e.g., "Good enough" or "Needs improvement").
    """
    def __init__(self, title: str, description: str, rating: Optional[str] = None):
        self.title = title
        self.description = description
        self.rating = rating

    def __repr__(self):
        return f"IdeaItem(title={self.title!r}, description={self.description!r}, rating={self.rating!r})"

def parse_ideas(json_ideas: list) -> List[IdeaItem]:
    """
    Convert a JSON list of idea dicts into a list of IdeaItem objects.
    
    Expected JSON format of each idea:
      {
        "title": "some title",
        "description": "some description"
      }

    Args:
        json_ideas (list): A list of dictionaries from the model response.

    Returns:
        list: A list of IdeaItem objects with title and description set.
    """
    idea_items = []
    for idea_dict in json_ideas:
        title = idea_dict.get("title", "")
        description = idea_dict.get("description", "")
        idea_items.append(IdeaItem(title, description))
    return idea_items

def ideas_to_json(ideas: List[IdeaItem]) -> list:
    """
    Convert a list of IdeaItem objects to a JSON-serializable list of dicts.

    Args:
        ideas (list): A list of IdeaItem objects.

    Returns:
        list: A list of dictionaries (title, description).
    """
    return [{"title": idea.title, "description": idea.description} for idea in ideas]

def evaluate_ideas(openai_client: OpenAIClient, ideas: List[IdeaItem]) -> List[bool]:
    """
    Send the ideas to the model for evaluation. Each idea gets a rating 
    ("Good enough" or "Needs improvement") and a reason.

    The function then updates each IdeaItem's 'rating' attribute 
    and returns a list of booleans indicating if each idea is "Good enough".

    Args:
        openai_client (OpenAIClient): The OpenAI client instance.
        ideas (list[IdeaItem]): A list of ideas to evaluate.

    Returns:
        list[bool]: For each idea, True if "Good enough", False if "Needs improvement".
    """
    system_message = "You are a helpful assistant that evaluates the quality of brainstorming ideas."
    # Convert IdeaItem objects to JSON for the prompt
    ideas_json = ideas_to_json(ideas)

    evaluation_prompt = f"""
    You are given a list of ideas, each with a "title" and a "description".
    For each idea, evaluate its creativity, practicality, and relevance.

    Return a JSON array of objects, each with:
    - "rating": "Good enough" or "Needs improvement"
    - "reason": A short explanation of why it got that rating

    Ideas:
    {json.dumps(ideas_json, ensure_ascii=False, indent=2)}
    """

    logging.info("Evaluating the quality of ideas...")
    evaluation_response = openai_client.query(system_message, evaluation_prompt)
    # Parse the response
    try:
        ratings_data = json.loads(clean_json_response(evaluation_response))
    except json.JSONDecodeError:
        logging.error("Failed to parse evaluation response. Marking all as Needs improvement.")
        # Mark all as "Needs improvement"
        for idea in ideas:
            idea.rating = "Needs improvement"
        return [False] * len(ideas)

    # ratings_data should be an array of objects with "rating" and "reason"
    bool_results = []
    for idea_item, rating_obj in zip(ideas, ratings_data):
        rating = rating_obj.get("rating", "Needs improvement")
        reason = rating_obj.get("reason", "")
        idea_item.rating = rating  # store the rating in the idea object

        is_good = (rating == "Good enough")
        bool_results.append(is_good)

        # Log out the reasoning for each idea
        logging.info(
            f"Idea: {idea_item.title}\n"
            f"Rating: {rating}\n"
            f"Reason: {reason}\n"
        )

    return bool_results

def clean_json_response(response: str) -> str:
    """
    Removes enclosing triple backticks and language hints (e.g., ```json) from a model response.

    Args:
        response (str): The raw string response from the model.

    Returns:
        str: The cleaned string, ready for JSON parsing.
    """
    if response.startswith("```json") and response.endswith("```"):
        return response[7:-3].strip()  # Remove ```json and ```
    elif response.startswith("```") and response.endswith("```"):
        return response[3:-3].strip()  # Remove ```
    return response

def generate_ideas(openai_client: OpenAIClient) -> List[IdeaItem]:
    """
    Generates 11 ideas, evaluates them, improves the weak ones, 
    and repeats until all are 'Good enough' or 5 attempts have passed.

    Returns:
        list[IdeaItem]: The final list of 11 ideas (each with title, description, rating).
    """
    system_message = "You are a helpful assistant that excels in generating ideas and brainstorming."
    question = "How might we use digital tools to help people build trust and express emotions in romantic relationships?"

    # Prompt to generate 11 ideas
    generation_prompt = f"""
    Generate 11 creative ideas in a JSON array to address the following question:
    "{question}"

    Each idea must be an object with:
    - "title": Short descriptive title
    - "description": A one or two sentence explanation of the idea

    Return exactly 11 objects in a valid JSON list, for example:
    [
      {{
        "title": "Idea Title 1",
        "description": "A short description."
      }},
      ...
    ]
    """

    max_attempts = 5
    attempts = 0

    # Separate lists for tracking ideas
    good_enough_ideas: List[IdeaItem] = []
    needs_improvement_ideas: List[IdeaItem] = []

    while attempts < max_attempts:
        attempts += 1
        logging.info(f"--- Attempt {attempts} of {max_attempts} ---")

        # 1) Generate initial ideas if no ideas exist
        if not good_enough_ideas and not needs_improvement_ideas:
            logging.info("Generating 11 initial ideas...")
            raw_generation_response = openai_client.query(system_message, generation_prompt)
            try:
                json_list = json.loads(clean_json_response(raw_generation_response))
                all_ideas = parse_ideas(json_list)
                needs_improvement_ideas.extend(all_ideas)
            except json.JSONDecodeError:
                logging.error("Failed to parse initial ideas. Retrying...")
                continue

        # 2) Evaluate the ideas in the `needs_improvement_ideas` list
        if needs_improvement_ideas:
            logging.info(f"Evaluating {len(needs_improvement_ideas)} ideas...")
            evaluation_results = evaluate_ideas(openai_client, needs_improvement_ideas)

            # Separate ideas into good enough and needs improvement
            for idx, is_good in enumerate(evaluation_results):
                idea = needs_improvement_ideas[idx]
                if is_good:
                    good_enough_ideas.append(idea)
                else:
                    idea.rating = "Needs improvement"

            # Update the `needs_improvement_ideas` list to only include weak ideas
            needs_improvement_ideas = [idea for idea in needs_improvement_ideas if idea.rating == "Needs improvement"]

        # 3) Improve the weak ideas if any remain
        if needs_improvement_ideas:
            logging.info(f"Improving {len(needs_improvement_ideas)} ideas...")
            weak_ideas_json = ideas_to_json(needs_improvement_ideas)
            good_ideas_json = ideas_to_json(good_enough_ideas)

            improvement_prompt = f"""
            You are given a list of ideas that need improvement to be more creative, practical, 
            and relevant. Each idea has "title" and "description". 

            You are also provided examples of ideas rated "Good enough". Use these examples 
            as inspiration when improving the weaker ideas.

            Good enough ideas:
            {json.dumps(good_ideas_json, ensure_ascii=False, indent=2)}

            Ideas to improve:
            {json.dumps(weak_ideas_json, ensure_ascii=False, indent=2)}

            Return the improved ideas in the same JSON structure.
            """

            improve_response = openai_client.query(system_message, improvement_prompt)
            try:
                improved_list = json.loads(clean_json_response(improve_response))
                improved_ideas = parse_ideas(improved_list)

                # Replace the weak ideas with the improved ones
                needs_improvement_ideas = improved_ideas
            except json.JSONDecodeError:
                logging.error("Failed to parse improved ideas. Retrying...")
                continue

        # 4) Check if all ideas are now good enough
        if not needs_improvement_ideas:
            logging.info("All ideas are Good enough!")
            return good_enough_ideas

    logging.warning("Reached maximum attempts. Some ideas may still need improvement.")
    return good_enough_ideas + needs_improvement_ideas

def categorize_ideas(openai_client: OpenAIClient, ideas: List[IdeaItem]) -> dict:
    """
    Categorize the ideas into 3–5 relevant themes using the model.

    Args:
        openai_client (OpenAIClient): The OpenAI client instance.
        ideas (list[IdeaItem]): A list of IdeaItem objects to categorize.

    Returns:
        dict: A dictionary where keys are theme names, and values are lists of IdeaItem titles.
    """
    system_message = "You are a helpful assistant that excels at grouping ideas into relevant themes."
    ideas_json = ideas_to_json(ideas)

    categorization_prompt = f"""
    You are given a list of ideas, each with a "title" and "description".
    Your task is to sort them into 3–5 relevant themes or categories.

    Return the result as a JSON object where the keys are the theme names, 
    and the values are lists of idea titles that belong to each theme.

    Ideas:
    {json.dumps(ideas_json, ensure_ascii=False, indent=2)}
    """

    logging.info("Categorizing ideas into themes...")
    categorization_response = openai_client.query(system_message, categorization_prompt)
    
    try:
        categories = json.loads(clean_json_response(categorization_response))
        logging.info(f"Categorization result:\n{json.dumps(categories, indent=2, ensure_ascii=False)}")
        return categories
    except json.JSONDecodeError:
        logging.error("Failed to parse categorization response. Returning an empty dictionary.")
        return {}

def main():
    openai_client = OpenAIClient()
    
    # Generate and improve ideas
    final_ideas = generate_ideas(openai_client)
    
    # Print final ideas
    print("Final Ideas:")
    for i, idea in enumerate(final_ideas, start=1):
        print(f"{i}. Title: {idea.title}")
        print(f"   Description: {idea.description}")
        print(f"   Rating: {idea.rating}")
        print("")

    # Categorize ideas into themes
    categories = categorize_ideas(openai_client, final_ideas)
    
    # Print categorized themes
    print("\nAffinity Diagram (Themes):")
    for theme, titles in categories.items():
        print(f"\nTheme: {theme}")
        for title in titles:
            print(f"- {title}")

if __name__ == "__main__":
    main()