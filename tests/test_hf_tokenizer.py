from ast import arg
from transformers import AutoTokenizer, AutoModel
import argparse
import os

SCRIPT_PATH=os.path.dirname(os.path.realpath(__file__))

def main(args):
    # tokenizer_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    if "/" in args.model_name:
        tokenizer_name = args.model_name
    elif "MiniLM" in args.model_name:
        tokenizer_name = f"sentence-transformers/{args.model_name}"
    elif "bge-" in args.model_name:
        tokenizer_name = f"BAAI/{args.model_name}"
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    with open(SCRIPT_PATH + "/test_prompts.txt", "r", encoding="utf-8") as f:
        inps = f.readlines()
        inps = list(map(lambda x: x.strip(), inps))

    print("Using tokenizer:", tokenizer_name)
    output = []
    for inp in inps:
        oup = tokenizer(inp, return_tensors="pt").input_ids[0].tolist()
        output.append(",".join([str(x) for x in oup]))
        for token in oup:
            print(f"{token} <--> {tokenizer.decode([token])}")

    with open(SCRIPT_PATH + "/hf_tokenized_ids.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download original repo files')
    parser.add_argument('model_name', type=str, help='Name of the repo')
    args = parser.parse_args()
    main(args)
