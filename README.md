# LLM Prefix Guidance

## 1) **Data**
- The finalized data from Llama-3-8B-instruct and Llama-3-70B-instruct can be found under "UPDATED_FINAL_DATA.csv".

- This data set includes ~2000 rows for each question and instruction used from the mix_instruct dataset

- The hints given to the small model are derived from the 50 tokens generated by the large model (e.g. 5_token_hint)

- "UPDATED_FINAL_DATA.csv" also contains corresponding small model output after being given the hint (e.g. 5_token_output)

- The LLM_full_output column is the full response of the LLM so we can compare this to the small model with the large model's hint



## 2) **Python Scripts**

- These are the Python scripts used to load the large and small model, generate responses, and write to csv files.

## 3) **SLURM Scripts**

- These scripts were used to request resources from the Compute Canada Narval cluster for the different jobs we required
