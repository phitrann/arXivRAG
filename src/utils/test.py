from pymongo import MongoClient
import os

os.environ["NO_PROXY"] = "localhost"

client = MongoClient("mongodb://admin:supersecret@localhost:27017")
db = client["chat_history"]
collection = db["test"]

item_1 = {
  "_id" : "U1IT00001",
  "item_name" : "Blender",
  "max_discount" : "10%",
  "batch_number" : "RR450020FRG",
  "price" : 340,
  "category" : "kitchen appliance"
}

item_2 = {
  "_id" : "U1IT00002",
  "item_name" : "Egg",
  "category" : "food",
  "quantity" : 12,
  "price" : 36,
  "item_description" : "brown country eggs"
}
# collection.insert_many([item_1,item_2])

item_details = collection.find()
for item in item_details:
   # This does not give a very readable output
   print(item)