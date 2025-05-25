import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CSV_IN = '/home/evotoole/large_llama/old_files/mix_instruct_2000.csv'
CSV_OUT = CSV_IN.replace('.csv', '_with_tkn_hints.csv')
MODEL_PATH = '/home/evotoole/scratch/llama3-70b-instruct'

num_tokens = [50]
batch_size = 3
save_after = 5

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
print("Tokenizer loaded:", type(tokenizer))

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model.eval()

def generate_with_llama_batch(prompts: list[str], num: int) -> list[str]:
    try:
        prompt_strs = prompts
        inputs = tokenizer(
            prompt_strs,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=num,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.3,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = []
        seq_len = inputs["input_ids"].shape[1]
        for i in range(len(prompts)):
            response_ids = outputs[i][seq_len:]
            decoded.append(tokenizer.decode(response_ids, skip_special_tokens=True).strip())
        return decoded

    except Exception as e:
        return [f"Error: {e}"] * len(prompts)


df = pd.read_csv(CSV_IN, header=None)
instructions = df.iloc[:, 1]
questions = df.iloc[:, 2]

hints_df = pd.DataFrame(index=range(len(questions)))
hints_df["Instruction"] = instructions
hints_df["Question"] = questions
for num in num_tokens:
    hints_df[f"output_{num}tkn_hint"] = ""

for num in num_tokens:
    for start_idx in tqdm(range(0, len(questions), batch_size), desc=f"Generating {num} token outputs"):
        end_idx = min(start_idx + batch_size, len(questions))
        batch_instructions = instructions[start_idx:end_idx]
        batch_questions = questions[start_idx:end_idx]

        batch_prompts = []
        for instr, ques in zip(batch_instructions, batch_questions):
            instr = str(instr).strip() if pd.notnull(instr) else ""
            ques = str(ques).strip() if pd.notnull(ques) else ""
            if instr and ques:
                prompt = f"### Instruction:\n{instr}\n\n### Input:\n{ques}\n\n### Response:"
            elif instr:
                prompt = f"### Instruction:\n{instr}\n\n### Response:"
            elif ques:
                prompt = f"### Instruction:\n{ques}\n\n### Response:"
            else:
                prompt = "### Instruction:\n\n### Response:"

            batch_prompts.append(prompt)

        batch_outputs = generate_with_llama_batch(batch_prompts, num)

        for i, output in enumerate(batch_outputs):
            row_idx = start_idx + i
            hints_df.at[row_idx, f"output_{num}tkn_hint"] = output

        if start_idx % save_after == 0:
            hints_df.to_csv(CSV_OUT, index=False, header=True)

hints_df.to_csv(CSV_OUT, index=False, header=True)
