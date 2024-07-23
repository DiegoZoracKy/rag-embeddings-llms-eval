import os
import re
import json
import yaml
from utils.hash_utils import generate_query_hash

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def extract_best_answer_id(response_text):
    match = re.search(r'ID da melhor resposta: (\S+)', response_text)
    return match.group(1) if match else None

def extract_groundedness_result(response_text):
    first_line = response_text.splitlines()[0].strip().upper()
    if "SIM" in first_line:
        return "SIM"
    elif "NÃO" in first_line:
        return "NÃO"
    else:
        return "UNKNOWN"

def read_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def process_evaluation_responses(evaluation_dir, experiments_dir):
    evaluation_results = {}

    experiment_dirs = [d for d in os.listdir(evaluation_dir) if os.path.isdir(os.path.join(evaluation_dir, d)) and d.startswith("experiment_")]

    for experiment_dir in experiment_dirs:
        experiment_path = os.path.join(evaluation_dir, experiment_dir)
        response_files = [f for f in os.listdir(experiment_path) if f.startswith("llm_response_") and f.endswith(".txt")]

        query_path = os.path.join(experiments_dir, experiment_dir, "query.txt")
        query = read_file(query_path) if os.path.exists(query_path) else "Query not found"

        evaluation_results[experiment_dir] = {
            "query": query,
            "evaluators": {},
            "runs": {}
        }

        for response_file in response_files:
            model = response_file.split("llm_response_")[1].split(".txt")[0]
            response_text = read_file(os.path.join(experiment_path, response_file))
            best_answer_id = extract_best_answer_id(response_text)

            if best_answer_id:
                evaluation_results[experiment_dir]["evaluators"][model] = {
                    "best_answer": best_answer_id
                }

        # Process experiment runs to include params
        run_dirs = [d for d in os.listdir(os.path.join(experiments_dir, experiment_dir)) if d.startswith("experiment_")]
        for run_dir in run_dirs:
            params_path = os.path.join(experiments_dir, experiment_dir, run_dir, "params.yaml")
            if os.path.exists(params_path):
                params = read_yaml(params_path)
                evaluation_results[experiment_dir]["runs"][run_dir] = params

            # Process groundedness evaluations
            run_hash = generate_query_hash(evaluation_results[experiment_dir]["query"])
            base_groundedness_path = os.path.join(experiment_path, run_dir, run_hash)
            if os.path.exists(base_groundedness_path):
                embedding_llm_dirs = [d for d in os.listdir(base_groundedness_path) if os.path.isdir(os.path.join(base_groundedness_path, d))]
                for embedding_llm_dir in embedding_llm_dirs:
                    groundedness_path = os.path.join(base_groundedness_path, embedding_llm_dir)
                    groundedness_files = [f for f in os.listdir(groundedness_path) if f.startswith("groundedness_evaluation_") and f.endswith(".txt")]
                    for groundedness_file in groundedness_files:
                        groundedness_model = groundedness_file.split("groundedness_evaluation_")[1].split(".txt")[0]
                        groundedness_text = read_file(os.path.join(groundedness_path, groundedness_file))
                        groundedness_result = extract_groundedness_result(groundedness_text)

                        if "groundedness_evaluators" not in evaluation_results[experiment_dir]["runs"][run_dir]:
                            evaluation_results[experiment_dir]["runs"][run_dir]["groundedness_evaluators"] = {}

                        evaluation_results[experiment_dir]["runs"][run_dir]["groundedness_evaluators"][groundedness_model] = {
                            "is_grounded": groundedness_result
                        }
                        # Debugging: Print content of groundedness files
                        print(f"Content of {groundedness_file}:")
                        print(groundedness_text)
                        print(f"Processed groundedness for {experiment_dir}, {run_dir}, {groundedness_model}: {groundedness_result}")  # Debugging statement

    return evaluation_results

evaluation_dir = "data/evaluations"
experiments_dir = "data/experiments"
evaluation_results = process_evaluation_responses(evaluation_dir, experiments_dir)

# Print or save the results as needed
print(json.dumps(evaluation_results, indent=4, ensure_ascii=False))

# Optionally, save the results to a JSON file
with open("evaluation_results_with_queries_and_params.json", "w", encoding='utf-8') as json_file:
    json.dump(evaluation_results, json_file, indent=4, ensure_ascii=False)
