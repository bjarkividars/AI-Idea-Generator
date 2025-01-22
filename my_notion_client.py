from notion_client import Client
import os
from dotenv import load_dotenv




class NotionClientWrapper:
    def __init__(self):
        """
        Initialize the Notion client.
        """
        load_dotenv()
        self.client = Client(auth=os.environ["NOTION_TOKEN"])
        self.PAGE_ID = os.environ["NOTION_PAGE_ID"]

    def append_custom_blocks_to_page(self, blocks: list):
        """
        Appends custom blocks to an existing Notion page.

        Args:
            blocks (list): A list of Notion block objects with specific formatting and content.
        """
        try:
            self.client.blocks.children.append(
                block_id=self.PAGE_ID,
                children=blocks
            )
            print(f"Added custom blocks to page: {self.PAGE_ID}")
        except Exception as e:
            print(f"Error while adding blocks: {e}")
            