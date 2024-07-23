experiments = {
    "experiment_1": {
        "openai": {
            "result": "openai_openai",
            "embedding_model": "openai",
            "llm_model": "openai"
        },
        "vertex": {
            "result": "vertex_openai",
            "embedding_model": "vertex",
            "llm_model": "openai"
        }
    },
    "experiment_2": {
        "openai": {
            "result": "vertex_vertex",
            "embedding_model": "vertex",
            "llm_model": "vertex"
        },
        "vertex": {
            "result": "openai_openai",
            "embedding_model": "openai",
            "llm_model": "openai"
        }
    },
    "experiment_3": {
        "openai": {
            "result": "openai_openai",
            "embedding_model": "openai",
            "llm_model": "openai"
        },
        "vertex": {
            "result": "vertex_openai",
            "embedding_model": "vertex",
            "llm_model": "openai"
        }
    },
    "experiment_4": {
        "openai": {
            "result": "vertex_openai",
            "embedding_model": "vertex",
            "llm_model": "openai"
        },
        "vertex": {
            "result": "vertex_openai",
            "embedding_model": "vertex",
            "llm_model": "openai"
        }
    },
    "experiment_5": {
        "openai": {
            "result": "openai_vertex",
            "embedding_model": "openai",
            "llm_model": "vertex"
        },
        "vertex": {
            "result": "openai_vertex",
            "embedding_model": "openai",
            "llm_model": "vertex"
        }
    },
    "experiment_6": {
        "openai": {
            "result": "openai_openai",
            "embedding_model": "openai",
            "llm_model": "openai"
        },
        "vertex": {
            "result": "openai_openai",
            "embedding_model": "openai",
            "llm_model": "openai"
        }
    },
    "experiment_7": {
        "openai": {
            "result": "vertex_openai",
            "embedding_model": "vertex",
            "llm_model": "openai"
        },
        "vertex": {
            "result": "vertex_openai",
            "embedding_model": "vertex",
            "llm_model": "openai"
        }
    },
    "experiment_8": {
        "openai": {
            "result": "vertex_openai",
            "embedding_model": "vertex",
            "llm_model": "openai"
        },
        "vertex": {
            "result": "openai_vertex",
            "embedding_model": "openai",
            "llm_model": "vertex"
        }
    },
    "experiment_9": {
        "openai": {
            "result": "openai_vertex",
            "embedding_model": "openai",
            "llm_model": "vertex"
        },
        "vertex": {
            "result": "vertex_openai",
            "embedding_model": "vertex",
            "llm_model": "openai"
        }
    },
    "experiment_10": {
        "openai": {
            "result": "vertex_openai",
            "embedding_model": "vertex",
            "llm_model": "openai"
        },
        "vertex": {
            "result": "vertex_openai",
            "embedding_model": "vertex",
            "llm_model": "openai"
        }
    }
}

# Initialize dictionaries to hold frequencies
embedding_model_freq = {"openai": 0, "vertex": 0}
llm_model_freq = {"openai": 0, "vertex": 0}
combined_freq = {}

# Iterate over each experiment and count frequencies
for experiment in experiments.values():
    for provider in experiment.values():
        embedding_model = provider["embedding_model"]
        llm_model = provider["llm_model"]
        combined = provider["result"]

        # Count embedding model
        embedding_model_freq[embedding_model] += 1

        # Count LLM model
        llm_model_freq[llm_model] += 1

        # Count combined result
        if combined not in combined_freq:
            combined_freq[combined] = 0
        combined_freq[combined] += 1

# Output the frequencies
print("Embedding Model Frequency:")
print(embedding_model_freq)

print("\nLLM Model Frequency:")
print(llm_model_freq)

print("\nCombined Model Frequency:")
print(combined_freq)
