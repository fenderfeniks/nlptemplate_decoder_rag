from prometheus_client import Counter, Histogram


# Счетчик генераций (разбиваем по источнику: tg_bot или rest_api)
LLM_GENERATIONS_TOTAL = Counter(
    "nlp_generations_total",
    "Total number of generation requests",
    ["source"],  # label: "tg", "rest"
)

# Время чистой работы ML-модели (без учета сети)
LLM_INFERENCE_TIME = Histogram(
    "nlp_inference_seconds", "Time spent generating response in LLM", ["source"]
)
