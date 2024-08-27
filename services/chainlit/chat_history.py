import os
from pymongo import MongoClient
import dotenv
from datetime import datetime

dotenv.load_dotenv()

class ChatHistory:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGODB_URI"))
        self.db = self.client.get_database("chat_database")
        self.collection = self.db.get_collection("chat_history")

    def save_chat(self, user_message, assistant_response):
        chat_entry = {
            "user_message": user_message,
            "assistant_response": assistant_response,
            "timestamp": datetime.utcnow()
        }
        self.collection.insert_one(chat_entry)

    def get_chat_history(self, limit=10):
        return list(self.collection.find().sort("_id", 1).limit(limit))

    def get_chats_by_date(self, date):
        start = datetime.combine(date, datetime.min.time())
        end = datetime.combine(date, datetime.max.time())
        return list(self.collection.find({"timestamp": {"$gte": start, "$lt": end}}))

    def delete_chat_history(self):
        self.collection.delete_many({})

    def count_messages(self):
        return self.collection.count_documents({})
    
    def format_chat_history(self, tokenizer, system_prompt, chats, message):
        chat_list = [{"role": "system", "content": system_prompt}]
        for chat in chats:
            chat_list.append({"role": "user", "content": chat["user_message"]})
            chat_list.append({"role": "assistant", "content": chat["assistant_response"]})
        chat_list.extend([{"role": "user", "content": message},
                          {"role": "assistant", "content": ""}])
        return tokenizer.apply_chat_template(chat_list, tokenize=False)