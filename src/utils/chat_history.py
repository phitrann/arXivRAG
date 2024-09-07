import os
from loguru import logger
from pymongo import MongoClient
from datetime import datetime, timezone

class ChatHistory:
    def __init__(self, uri=None, db_name=None, collection_name=None, tokenizer=None, top_k=2):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.top_k = top_k

        self.tokenizer = tokenizer
        self.client = MongoClient(self.uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]

    def save_chat(self, user_message, assistant_response):
        chat_entry = {
            "user_message": user_message,
            "assistant_response": assistant_response,
            "timestamp": datetime.now(timezone.utc)
        }
        self.collection.insert_one(chat_entry)

    def get_chat_history(self) :
        return list(self.collection.find().sort("_id", 1).limit(self.top_k))

    def get_chats_by_date(self, date):
        start = datetime.combine(date, datetime.min.time())
        end = datetime.combine(date, datetime.max.time())
        return list(self.collection.find({"timestamp": {"$gte": start, "$lt": end}}))

    def delete_chat_history(self):
        self.collection.delete_many({})

    def count_messages(self):
        return self.collection.count_documents({})
    
    def format_chat_history(self, system_prompt, chats, message):
        chat_list = [{"role": "system", "content": system_prompt}]
        logger.info(f"{chats = }")
        for chat in chats:
            chat_list.append({"role": "user", "content": chat["user_message"]})
            chat_list.append({"role": "assistant", "content": chat["assistant_response"]})
        chat_list.extend([{"role": "user", "content": message},
                          {"role": "assistant", "content": ""}])
        return self.tokenizer.apply_chat_template(chat_list, tokenize=False)