import json
import argparse
from typing import Any, Dict, List, Union

def pick_assistant_text(arr: Union[List[Any], Dict[Any, Any]]) -> str:
    items = arr.values() if isinstance(arr, dict) else arr
    for x in items:
        if isinstance(x, dict) and x.get("role") == "assistant":
            return x.get("content") or x.get("value") or ""
    return ""

def transform_record(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id":            item.get("id"),
        "source":        item.get("source"),
        "prompt":        item.get("prompt"),
        "features_used": item.get("features_used"),
        "is_swapped":    item.get("is_swapped"),
        "highest_level_degree": item.get("highest_level_degree"),
        "conversations": [
            {"from": "human", "value": item.get("prompt", "")}
        ],
        "chosen": {
            "from":  "gpt",
            "value": pick_assistant_text(item.get("chosen", []))
        },
        "rejected": {
            "from":  "gpt",
            "value": pick_assistant_text(item.get("rejected", []))
        }
    }

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

def main(input_path: str, output_path: str):
    # 读取 JSONL
    data = read_jsonl(input_path)
    # 转换
    transformed = [transform_record(rec) for rec in data]
    # 输出为 JSON array
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transformed, f, ensure_ascii=False, indent=2)
    print(f"✅ 转换完成 → {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="读取 JSONL 并转换成指定格式 JSON")
    parser.add_argument("input",  help="输入 JSONL 文件路径")
    parser.add_argument("output", help="输出 JSON 文件路径")
    args = parser.parse_args()
    main(args.input, args.output)
