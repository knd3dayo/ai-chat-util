# opencode.jsonを生成して第１引数のパスに保存するスクリプト
import json
import os
import sys

def main(output_path: str):
    # 環境変数からLLMの設定を取得
    llm_api_key = os.getenv("LLM_API_KEY" )
    if not llm_api_key:
        raise ValueError("LLM_API_KEY environment variable is required")
    llm_provider = os.getenv("LLM_PROVIDER")
    if not llm_provider:
        raise ValueError("LLM_PROVIDER environment variable is required")
    llm_model = os.getenv("LLM_MODEL")
    if not llm_model:
        raise ValueError("LLM_MODEL environment variable is required")

    # Prefer a container-reachable base URL when provided.
    llm_base_url = os.getenv("LLM_BASE_URL_IN_CONTAINER") or os.getenv("LLM_BASE_URL")

    config = {}

    provider_options= {"apiKey": llm_api_key}
    if llm_base_url:
        provider_options["baseURL"] = llm_base_url

    config["$schema"] = "https://opencode.ai/config.json"
    config["provider"] = {
        llm_provider: {
            "options": provider_options
        }
    }
    config["model"] = f"{llm_provider}/{llm_model}"

    """
    config["mcp"] = {}
    """


    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Generated opencode.json at {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Output path argument is required")
    output_path = sys.argv[1]
    main(output_path)

