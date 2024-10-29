import json
import random

def create_small_data(src_json):
    with open(src_json) as f:
        lists = json.loads(f.read())
    
    random.shuffle(lists)
    with open("gobjaverse/gobjaverse_small.json", 'w') as f1:
        f1.write(json.dumps(lists[:10], indent=2))


create_small_data("gobjaverse/gobj_merged.json")