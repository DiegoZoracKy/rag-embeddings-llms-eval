# src/app/llm/run_prepare_prompts.py

import os

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def read_all_evaluation_prompts(evaluation_dir):
    evaluation_dirs = [d for d in os.listdir(evaluation_dir) if os.path.isdir(os.path.join(evaluation_dir, d)) and d.startswith("experiment_")]
    all_prompts = []

    for experiment_dir in evaluation_dirs:
        prompt_file = os.path.join(evaluation_dir, experiment_dir, "evaluation_prompt.txt")
        if os.path.exists(prompt_file):
            prompt_text = read_file(prompt_file)
            all_prompts.append({
                "experiment_dir": experiment_dir,
                "prompt": prompt_text
            })

    return all_prompts

def prepare_prompt(base_dir, experiment_dir):
    # Read the main query
    query_file = os.path.join(base_dir, "query.txt")
    query_text = read_file(query_file)

    prompt = f"Questão (id: {experiment_dir}):\n{query_text}\n\n"
    run_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("experiment_")]

    response_id = 1
    for run_dir in run_dirs:
        run_path = os.path.join(base_dir, run_dir)
        sub_dirs = [d for d in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, d))]

        for sub_dir in sub_dirs:
            sub_path = os.path.join(run_path, sub_dir)
            inner_dirs = [d for d in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, d))]

            for inner_dir in inner_dirs:
                inner_path = os.path.join(sub_path, inner_dir)
                response_files = [f for f in os.listdir(inner_path) if f.startswith("llm_response_") and f.endswith(".txt")]

                for response_file in response_files:
                    response_text = read_file(os.path.join(inner_path, response_file))
                    prompt += f"Resposta {response_id} (id: {run_dir}):\n{response_text}\n\n"
                    response_id += 1

    return prompt

def process_experiments(data_dir):
    experiment_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("experiment_")]

    for experiment_dir in experiment_dirs:
        base_dir = os.path.join(data_dir, experiment_dir)
        prompt = prepare_prompt(base_dir, experiment_dir)

        evaluation_dir = os.path.join("data", "evaluations", experiment_dir)
        os.makedirs(evaluation_dir, exist_ok=True)

        prompt_file = os.path.join(evaluation_dir, "evaluation_prompt.txt")
        with open(prompt_file, 'w', encoding='utf-8') as file:
            file.write(prompt)

        print(f"Prompt saved to {prompt_file}")

def prepare_groundedness_prompt(prompt_file, response_file):
    context = read_file(prompt_file)
    response = read_file(response_file)

    evaluation_prompt = (
        f"Instruções de Avaliação:\n\n"
        f"Verifique se a resposta fornecida pelo LLM está fundamentada no contexto fornecido. O principal objetivo é garantir que a resposta (seção '1. Resposta') não inclua informações não presentes no contexto (Contexto). Caso a resposta inclua informações que não estão no contexto, deve ser marcada como 'NÃO'. Caso contrário, marque 'SIM'. Após isso, avalie os seguintes pontos:\n\n"
        f"1. Corretude: A resposta fornecida (seção '1. Resposta') está correta com base no conteúdo do contexto?\n"
        f"2. Completude: A explicação fornecida (seção '2. Explicação') é completa e justifica adequadamente a resposta?\n"
        f"3. Referência: As referências fornecidas (seção '3. Referência') estão diretamente relacionadas e são retiradas corretamente do contexto?\n\n"
        f"Primeiro, escreva 'SIM' ou 'NÃO' para indicar se a resposta está fundamentada no contexto, e depois forneça os detalhes da avaliação.\n\n"
        f"Prompt para o LLM:\n{context}\n\n"
        f"Resposta do LLM:\n{response}"
    )
    return evaluation_prompt

def process_experiments_for_groundedness(data_dir):
    experiment_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("experiment_")]

    for experiment_dir in experiment_dirs:
        base_dir = os.path.join(data_dir, experiment_dir)

        run_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("experiment_")]

        for run_dir in run_dirs:
            run_path = os.path.join(base_dir, run_dir)
            sub_dirs = [d for d in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, d))]

            for sub_dir in sub_dirs:
                sub_path = os.path.join(run_path, sub_dir)
                inner_dirs = [d for d in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, d))]

                for inner_dir in inner_dirs:
                    inner_path = os.path.join(sub_path, inner_dir)
                    prompt_files = [f for f in os.listdir(inner_path) if f.startswith("prompt_") and f.endswith(".txt")]
                    
                    for prompt_file in prompt_files:
                        model = prompt_file.split("prompt_")[1].split(".txt")[0]
                        response_file = os.path.join(inner_path, f"llm_response_{model}.txt")
                        if os.path.exists(response_file):
                            evaluation_prompt = prepare_groundedness_prompt(os.path.join(inner_path, prompt_file), response_file)
                            evaluation_dir = os.path.join("data", "evaluations", experiment_dir, run_dir, sub_dir, inner_dir)
                            os.makedirs(evaluation_dir, exist_ok=True)
                            groundedness_prompt_file = os.path.join(evaluation_dir, f"groundedness_prompt_{model}.txt")
                            with open(groundedness_prompt_file, 'w', encoding='utf-8') as file:
                                file.write(evaluation_prompt)
                            print(f"Groundedness prompt saved to {groundedness_prompt_file}")
