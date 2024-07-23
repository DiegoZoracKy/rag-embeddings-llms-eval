import json
from collections import defaultdict
from typing import Any, Dict

def compute_intermediate_frequencies(evaluation_results):
    frequency = defaultdict(lambda: {"llm_model": defaultdict(int), "embeddings_model": defaultdict(int), "combined_model": defaultdict(int)})

    for experiment_id, experiment_data in evaluation_results.items():
        evaluators = experiment_data.get("evaluators", {})
        runs = experiment_data.get("runs", {})
        
        for evaluator, evaluator_data in evaluators.items():
            best_answer_id = evaluator_data.get("best_answer")
            if best_answer_id in runs:
                best_run_data = runs[best_answer_id]
                llm_model = best_run_data.get("llm_model")
                embeddings_model = best_run_data.get("embeddings", {}).get("model")
                
                if llm_model:
                    llm_model_str = f"{llm_model['provider']}_{llm_model['model']}"
                    frequency[experiment_id]["llm_model"][llm_model_str] += 1
                if embeddings_model:
                    frequency[experiment_id]["embeddings_model"][embeddings_model] += 1
                if llm_model and embeddings_model:
                    combined_model_str = f"{embeddings_model}_{llm_model['provider']}_{llm_model['model']}"
                    frequency[experiment_id]["combined_model"][combined_model_str] += 1

    return frequency

def compute_final_results(frequency_results):
    final_result = {
        "llm_model": defaultdict(int),
        "embeddings_model": defaultdict(int),
        "combined_model": defaultdict(int)
    }

    for experiment_id, models in frequency_results.items():
        for llm_model, count in models["llm_model"].items():
            final_result["llm_model"][llm_model] += count
        for embeddings_model, count in models["embeddings_model"].items():
            final_result["embeddings_model"][embeddings_model] += count
        for combined_model, count in models["combined_model"].items():
            final_result["combined_model"][combined_model] += count

    return final_result

def compute_groundedness_intermediate_frequencies(evaluation_results):
    groundedness_frequency = defaultdict(lambda: defaultdict(int))

    for experiment_id, experiment_data in evaluation_results.items():
        runs = experiment_data.get("runs", {})
        
        for run_id, run_data in runs.items():
            llm_model = run_data.get("llm_model")
            embeddings_model = run_data.get("embeddings", {}).get("model")
            if llm_model and embeddings_model:
                combined_model_str = f"{embeddings_model}_{llm_model['provider']}_{llm_model['model']}"
                groundedness_evaluators = run_data.get("groundedness_evaluators", {})
                
                for evaluator, evaluator_data in groundedness_evaluators.items():
                    is_grounded = evaluator_data.get("is_grounded")
                    if is_grounded == "SIM":
                        groundedness_frequency[experiment_id][combined_model_str] += 1

    return groundedness_frequency

def compute_groundedness_final_results(groundedness_frequency_results):
    groundedness_final_result = defaultdict(int)

    for experiment_id, models in groundedness_frequency_results.items():
        for model, count in models.items():
            groundedness_final_result[model] += count

    return groundedness_final_result

# Load the evaluation results from the JSON file
with open("evaluation_results_with_queries_and_params.json", "r", encoding='utf-8') as json_file:
    evaluation_results = json.load(json_file)

# Compute the intermediate frequencies
frequency_results = compute_intermediate_frequencies(evaluation_results)
groundedness_frequency_results = compute_groundedness_intermediate_frequencies(evaluation_results)

# Compute the final results
final_result = compute_final_results(frequency_results)
groundedness_final_result = compute_groundedness_final_results(groundedness_frequency_results)

# Function to save the results to a file
def save_results_to_file(file_path: str, results: Dict[str, Any]):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("Final Result for Evaluators:\n")
        f.write(json.dumps(final_result, indent=4, ensure_ascii=False))
        f.write("\n\nFinal Result for Groundedness Evaluators:\n")
        f.write(json.dumps(groundedness_final_result, indent=4, ensure_ascii=False))

# Assuming your script already computes the following variables:
# final_result, groundedness_final_result

# Save the results to a file
output_file_path = "data/evaluations/computed_results.txt"
save_results_to_file(output_file_path, {
    "final_result": final_result,
    "groundedness_final_result": groundedness_final_result
})

# Print the results
print("Final Result for Evaluators:")
print(json.dumps(final_result, indent=4, ensure_ascii=False))
print("\nFinal Result for Groundedness Evaluators:")
print(json.dumps(groundedness_final_result, indent=4, ensure_ascii=False))
