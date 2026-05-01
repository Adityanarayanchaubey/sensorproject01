from pymongo.mongo_client import MongoClient

import pandas as pd
import json

uri="mongodb+srv://aditya:12345@cluster0.vk14avj.mongodb.net/?appName=Cluster0"

#create a new c,ient and connect to server
client=MongoClient(uri)

#create a database name and collection name
DATABASE_NAME="pwskills"
COLLECTION_NAME='waferfault'

df=pd.read_csv("E:\Sensor Project\notebooks\wafer_23012020_041211.csv")

df=df.drop('Unnamed: 0',axis=1)

json_record=list(json.loads(df.T.to_json()).values())

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record) 