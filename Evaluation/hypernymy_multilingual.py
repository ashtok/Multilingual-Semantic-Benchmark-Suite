from lm_eval.api.task import Task
import json

class multilingual_qa_task(Task):
    VERSION = 0

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_docs(self):
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def doc_to_text(self, doc):
        options = "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(doc["options"]))
        return f"{doc['prompt']}\n{options}\nAnswer:"

    def doc_to_target(self, doc):
        return " " + doc["options"][doc["answer_index"]]
