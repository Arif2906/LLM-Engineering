from pydantic import BaseModel
from typing import Optional, Any
import os
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from itertools import repeat
from datasets import Dataset, DatasetDict, load_dataset
from typing import Self


class Item(BaseModel):
    title: str
    price: float
    description: Optional[str] = None
    category: str
    id: Optional[int] = None
    

    @classmethod
    def parse_data(cls, datapoint,category):
        if datapoint is None:
            return None
        try:
            price= float(datapoint["price"])
        except ValueError:
            return None
        title= normalize_text(datapoint["title"])
        if not isinstance(title, str):
            return None
        description = (normalize_text(datapoint.get("description")) or "")[:100] or None
        if not isinstance(description, str):
            return None 
        if not 1<= price <= 600:
            return None
        else:
            return cls(
                title=title,
                price=price,
                description=description,
                category=category
            )
    @classmethod
    def push_to_hub(cls,dataset_name,train_dataset:list[Self],validation_dataset:list[Self],test_dataset:list[Self]):
        full = f"{dataset_name}"
        DatasetDict({"train":Dataset.from_list([item.model_dump() for item in train_dataset]),
                     "validation":Dataset.from_list([item.model_dump() for item in validation_dataset]) ,
                     "test":Dataset.from_list([item.model_dump() for item in test_dataset])}).push_to_hub(full)
class ItemLoader:
    def __init__(self, category):
        self.category = category
    cpu_count = os.cpu_count()
    worker_count = cpu_count - 1 if cpu_count else 1

    def load_multiple(self):
        items= []
        dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{self.category}", trust_remote_code=True, split="full")
        with ThreadPoolExecutor(max_workers=self.worker_count) as executor:
            items = list(executor.map(Item.parse_data, dataset, repeat(self.category)))
        items = [item for item in items if item is not None]
        return items



def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    
    if isinstance(value, str):
        value = value.strip()
        return value if value else None

    if isinstance(value, list):
        parts = [normalize_text(v) for v in value]
        parts = [p for p in parts if p]
        return " ".join(parts) if parts else None

    if isinstance(value, dict):
        parts = [normalize_text(v) for v in value.values()]
        parts = [p for p in parts if p]
        return " ".join(parts) if parts else None

    return str(value).strip()