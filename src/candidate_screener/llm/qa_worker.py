"""Local offline JSON-lines extractive QA worker; never prints private errors."""
import argparse
import json
import sys


def main():
    from .extractive_qa import ExtractiveQA
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    try:
        model = ExtractiveQA(args.manifest)
    except Exception:
        model = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            answer = model.answer(request["question"], request["passages"]) if model else {"status": "unavailable", "spans": [], "exhaustive": False}
        except Exception:
            answer = {"status": "unavailable", "spans": [], "exhaustive": False}
        print(json.dumps(answer), flush=True)


if __name__ == "__main__":
    main()
