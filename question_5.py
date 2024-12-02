import re
from collections import defaultdict
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def connect_to_database(db_name):
    DB_NAME = db_name
    DB_HOST = "localhost"
    DB_PORT = 27017
    try:
        client = MongoClient(host=DB_HOST, port=DB_PORT)
        db = client[DB_NAME]
        return db
    except:
        print("Database did not connect successfully")

def clean_text(original_text):
    removed_punctuation = re.sub(r'[^\w\s]', '', original_text)
    return removed_punctuation.lower()


documents = ["After the medication, headache and nausea were reported by the patient.",
             "The patient reported nausea and dizziness caused by the medication.",
             "Headache and dizziness are common effects of this medication.",
             "The medication caused a headache and nausea, but no dizziness was reported."]


db = connect_to_database("assignment_4")
corpus_collection = db["documents"]
for text in documents:
    doc = dict()
    doc["content"] = text
    corpus_collection.insert_one(doc)


master_doc_collection = list()
master_doc_ids = list()
for doc in corpus_collection.find():
    master_doc_collection.append(doc)
    master_doc_ids.append(doc["_id"])


cleaned_source_docs = list()
for text in master_doc_collection:
    cleaned_text = clean_text(text["content"])
    cleaned_source_docs.append(cleaned_text)


vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3))
tfidf_matrix = vectorizer.fit_transform(cleaned_source_docs)
feature_names = vectorizer.get_feature_names_out()
idf_values = vectorizer.idf_
term_idf_map = dict(zip(feature_names, idf_values))
terms_collection = db["terms"]
inverted_index = defaultdict(list)
for doc_idx, doc_id in enumerate(master_doc_ids):
    doc_vector = tfidf_matrix.getrow(doc_idx).tocoo()

    for term_idx, tfidf_score in zip(doc_vector.col, doc_vector.data):
        term = feature_names[term_idx]

        inverted_index[term].append({
            "doc_id": doc_id,
            "tfidf": tfidf_score,
        })


for term, entries in inverted_index.items():
    entry = dict()
    entry["term"] = term
    entry["pos"] = vectorizer.vocabulary_[term]
    entry["idf"] = term_idf_map[term]
    entry["documents"] = entries
    terms_collection.insert_one(entry)


q1 = "nausea and dizziness"
q2 = "effects"
q3 = "nausea was reported"
q4 = "dizziness"
q5 = "the medication"

