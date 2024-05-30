unset http_proxy
unset https_proxy
time curl -H "Host: llmtorch-predictor.default.svc.cluster.local" -H "Content-Type: application/json" http://172.16.19.82/v1/models/llm:predict -d @./input.json
