import os
import json
import csv
from datetime import datetime

class Logger:
    def __init__(self, path: str = "logs"):
        os.makedirs(path, exist_ok=True)
        self.csv_file = os.path.join(path, "queries.csv")
        self.json_file = os.path.join(path, "queries.json")
        # ensure CSV header
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["group_id", "timestamp", "question", "retrieved_chunks", "prompt", "generated_answer"])
        # ensure JSON file exists with empty list
        if not os.path.exists(self.json_file):
            with open(self.json_file, "w", encoding='utf-8') as f:
                json.dump([], f)

    def log(self, question: str, retrieved: list, prompt: str, answer: str):
        entry = {
            "group_id": "Team Neural Narrators",
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "retrieved_chunks": retrieved,
            "prompt": prompt,
            "generated_answer": answer,
        }
        # append to JSON array
        with open(self.json_file, "r+", encoding='utf-8') as f:
            data = json.load(f)
            data.append(entry)
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.truncate()

        # append to CSV as before
        with open(self.csv_file, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                entry['group_id'],
                entry['timestamp'],
                question,
                json.dumps(retrieved, ensure_ascii=False),
                prompt,
                answer,
            ])
