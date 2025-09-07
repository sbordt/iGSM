from data_gen.pretrain.id_gen import IdGen
from tools.tools import tokenizer, fix_seed
from typing import Literal
from tqdm import tqdm
import json

def get_prob_sol_ans_triple(tpy: Literal["med", "hard"]):
    assert tpy in ["med", "hard"], "Invalid type: Choose 'med' or 'hard'"
    # Set parameters based on difficulty
    max_op = 15 if tpy == "med" else 21
    max_edge = 20 if tpy == "med" else 28

    id_gen = IdGen(
        max_op=max_op,        # Maximum # of operations
        max_edge=max_edge,    # Maximum # of edges (instance parameters) in the structure graph
        perm_level=5,         # Random shuffle level for problem description
        detail_level=0        # Most detailed solution format
    )

    id_gen.gen_prob([i for i in range(23)], p_format="pq")

    return id_gen
    
problem_list = []
id_gen_list = []
fix_seed(42)
for idx in tqdm(range(1000)):
    id_gen = get_prob_sol_ans_triple("med")
    problem_list.append({
        "problem": tokenizer.decode(id_gen.prob_token).strip(),
        "solution": tokenizer.decode(id_gen.sol_token).strip(),
        "answer": tokenizer.decode(id_gen.ans_token).strip(),
        "op": id_gen.op_
    })
    id_gen_list.append(id_gen)

import pickle
with open("iGSM.pkl", "wb") as f:
    pickle.dump(id_gen_list, f)

with open(f"iGSM.jsonl", "w") as f:
    for item in problem_list:
        f.write(json.dumps(item) + '\n')

