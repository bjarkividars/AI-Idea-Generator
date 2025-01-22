from openai import OpenAI
import os
import logging
from dotenv import load_dotenv
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class OpenAIClient:
    def __init__(self, model="gpt-4o"):
        """
        Initialize the OpenAI client with a specified model.
        """
        load_dotenv()
        
        # OpenAI.api_key = os.environ.get("OPENAI_KEY")
        # print(os.environ["OPENAI_KEY"])
        self.client = OpenAI(api_key=os.environ["OPENAI_KEY"])
        self.model = model

    def query(self, system_message: str, user_message: str) -> str:
        """
        Query the OpenAI model with a system message and a user message.

        Args:
            system_message (str): The role of the assistant, e.g., its behavior or capabilities.
            user_message (str): The prompt or query provided by the user.

        Returns:
            str: The response content from the assistant.
        """
        # Log the prompts being sent to the model
        logging.info(f"System Message: {system_message}")
        logging.info(f"User Message: {user_message}")

        # Query the OpenAI API
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        )

        # Extract and log the response
        response = completion.choices[0].message.content
        logging.info(f"Model Response: {response}")
        if response is None:
            raise Exception("OpenAI API returned None")
        return response