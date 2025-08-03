import multiprocessing as mp
import json
import os
from typing import Literal
from tqdm import tqdm
import argparse

from data_gen.pretrain.id_gen import IdGen
from tools.tools import tokenizer, fix_seed


def get_prob_sol_ans_triple(tpy: Literal["med", "hard"]):
    """Create and configure IdGen based on difficulty type"""
    assert tpy in ["med", "hard"], "Invalid type: Choose 'med' or 'hard'"
    
    # Set parameters based on difficulty
    max_op = 15 if tpy == "med" else 21
    max_edge = 20 if tpy == "med" else 28
    
    id_gen = IdGen(
        max_op=max_op,  # Maximum # of operations
        max_edge=max_edge,  # Maximum # of edges (instance parameters) in the structure graph
        perm_level=5,  # Random shuffle level for problem description
        detail_level=0  # Most detailed solution format
    )
    
    id_gen.gen_prob([i for i in range(23)], p_format="pq")
    return id_gen


def worker_process(process_id: int, num_problems: int, base_seed: int, difficulty: str, output_dir: str):
    """
    Worker process that generates a subset of problems and saves them to a temporary file
    
    Args:
        process_id: Unique identifier for this process
        num_problems: Number of problems this process should generate
        base_seed: Base seed for reproducibility
        difficulty: Problem difficulty ("med" or "hard")
        output_dir: Directory to save temporary files
    """
    # Set unique seed for this process
    process_seed = base_seed + process_id * 1000000  # Ensure seeds don't overlap
    fix_seed(process_seed)
    
    problem_list = []
    
    # Generate problems with progress bar for this process
    for i in tqdm(range(num_problems), desc=f"Process {process_id}", position=process_id):
        id_gen = get_prob_sol_ans_triple(difficulty)
        problem_list.append({
            "problem": tokenizer.decode(id_gen.prob_token).strip(),
            "solution": tokenizer.decode(id_gen.sol_token).strip(),
            "answer": tokenizer.decode(id_gen.ans_token).strip(),
            "op": id_gen.op_
        })
    
    # Save to temporary file
    temp_filename = os.path.join(output_dir, f"temp_problems_{process_id}.jsonl")
    with open(temp_filename, "w") as f:
        for item in problem_list:
            f.write(json.dumps(item) + '\n')
    
    print(f"Process {process_id} completed: {num_problems} problems saved to {temp_filename}")
    return temp_filename


def combine_results(temp_files: list, output_filename: str, cleanup: bool = True):
    """
    Combine results from all worker processes into a single file
    
    Args:
        temp_files: List of temporary file paths
        output_filename: Final output filename
        cleanup: Whether to delete temporary files after combining
    """
    print("Combining results from all processes...")
    
    total_problems = 0
    with open(output_filename, "w") as outfile:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                with open(temp_file, "r") as infile:
                    for line in infile:
                        outfile.write(line)
                        total_problems += 1
                
                # Clean up temporary file if requested
                if cleanup:
                    os.remove(temp_file)
            else:
                print(f"Warning: Temporary file {temp_file} not found")
    
    print(f"Combined {total_problems} problems into {output_filename}")


def generate_problems_multiprocess(total_problems: int, global_seed: int, difficulty: str = "med", 
                                 output_filename: str = "iGSM.jsonl", num_processes: int = None):
    """
    Main function to generate problems using multiprocessing
    
    Args:
        total_problems: Total number of problems to generate
        global_seed: Global seed for reproducibility
        difficulty: Problem difficulty ("med" or "hard")
        output_filename: Final output filename
        num_processes: Number of processes to use (default: CPU count)
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print(f"Generating {total_problems} problems using {num_processes} processes")
    print(f"Difficulty: {difficulty}, Global seed: {global_seed}")
    
    # Create temporary directory for intermediate files
    temp_dir = "temp_problem_gen"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Calculate problems per process
    problems_per_process = total_problems // num_processes
    remaining_problems = total_problems % num_processes
    
    # Create process arguments
    process_args = []
    for i in range(num_processes):
        # Last process gets any remaining problems
        num_problems = problems_per_process + (1 if i < remaining_problems else 0)
        process_args.append((i, num_problems, global_seed, difficulty, temp_dir))
    
    # Start multiprocessing
    print("Starting worker processes...")
    with mp.Pool(processes=num_processes) as pool:
        # Use starmap to unpack arguments
        temp_files = pool.starmap(worker_process, process_args)
    
    # Combine all results
    combine_results(temp_files, output_filename, cleanup=True)
    
    # Clean up temporary directory
    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
        os.rmdir(temp_dir)
    
    print(f"Generation complete! Results saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate problems using multiprocessing")
    parser.add_argument("--total_problems", type=int, default=1000000, 
                       help="Total number of problems to generate")
    parser.add_argument("--global_seed", type=int, default=42, 
                       help="Global seed for reproducibility")
    parser.add_argument("--difficulty", type=str, default="med", choices=["med", "hard"],
                       help="Problem difficulty")
    parser.add_argument("--output", type=str, default="iGSM.jsonl",
                       help="Output filename")
    parser.add_argument("--num_processes", type=int, default=None,
                       help="Number of processes to use (default: CPU count)")
    
    args = parser.parse_args()
    
    generate_problems_multiprocess(
        total_problems=args.total_problems,
        global_seed=args.global_seed,
        difficulty=args.difficulty,
        output_filename=args.output,
        num_processes=args.num_processes
    )






    