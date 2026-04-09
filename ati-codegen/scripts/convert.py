import json

def convert(problem):
    task_id = f"{problem['id']}_{problem['title'].lower().replace(' ','_')}"
    
    entry_point = "min_days"

    prompt = f"""def {entry_point}(power):
    \"\"\"
{problem['content']['problem']}
    \"\"\"
"""

    tests = []
    for ex in problem["example"]:
        tests.append(f"assert {entry_point}({ex['input']}) == {ex['output']}")

    test_code = "\n".join(tests)

    return {
        "task_id": task_id,
        "entry_point": entry_point,
        "prompt": prompt,
        "test": test_code
    }

with open("input.json") as f:
    data = json.load(f)

converted = convert(data)

with open("passk_dataset.json", "w") as f:
    json.dump(converted, f, indent=2)