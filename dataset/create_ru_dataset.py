import pandas as pd
import numpy as np
from datasets import load_dataset
np.random.seed(42)


df_alpaca = load_dataset('d0rj/alpaca-cleaned-ru')

df_alpaca = df_alpaca["train"].to_pandas()
df_alpaca["input"] = df_alpaca["instruction"]+df_alpaca["input"]
df_alpaca=df_alpaca.drop("instruction",axis=1).sample(15000)

df_gptj = load_dataset('d0rj/synthetic-instruct-gptj-pairwise-ru')
df_gptj = df_gptj["train"].to_pandas()
df_gptj = df_gptj.rename(columns = {"prompt":"input", "chosen":"output"}).drop("rejected",axis=1).sample(9000)


df_boolq = load_dataset('d0rj/boolq-ru')
df_boolq = df_boolq["train"].to_pandas()
df_boolq =df_boolq.rename(columns={"question":"input","passage":"output"}).sample(3000).drop("answer",axis=1)

full_df = pd.concat([df_alpaca,df_boolq,df_gptj])
full_df = full_df.drop_duplicates(subset=["input"])
print(full_df.head(10))
full_df.to_csv("full_sft_26k.csv", index= False)