#!/usr/bin/env python3
import subprocess
import requests
import time
import json
import os
import matplotlib.pyplot as plt

# --- Stop all running Ollama models ---
def stop_running_models():
    stop_command = r"ollama ps | awk 'NR>1 {print $1}' | xargs -L 1 -I {} ollama stop {}"
    print("Stopping running Ollama models (if any)...")
    subprocess.run(stop_command, shell=True)
    print("Done.\n")

# --- Approximate token count function ---
def calculate_num_tokens(messages):
    # Rough approximation: 1 token ~ 4 characters
    num_tokens = 0
    for message in messages:
        num_tokens += len(message['content']) / 4
    return int(num_tokens)

# --- Message builders ---
def get_default_messages(article_text):
    system_instructions = """
A chat between a curious user and an artificial intelligence Assistant. The Assistant is an expert at identifying entities and relationships in text. The Assistant responds in JSON output only.

The User provides text in the format:

-------Text begin-------
<User provided text>
-------Text end-------

The Assistant follows the following steps before replying to the User:

1. **identify the most important entities** The Assistant identifies the most important entities in the text. These entities are listed in the JSON output under the key "nodes", they follow the structure of a list of dictionaries where each dict is:

"nodes":[{"id": <entity N>, "type": <type>, "detailed_type": <detailed type>}, ...]

where "type": <type> is a broad categorization of the entity. "detailed type": <detailed_type>  is a very descriptive categorization of the entity.

2. **determine relationships** The Assistant uses the text between -------Text begin------- and -------Text end------- to determine the relationships between the entities identified in the "nodes" list defined above. These relationships are called "edges" and they follow the structure of:

"edges":[{"from": <entity 1>, "to": <entity 2>, "label": <relationship>}, ...]

The <entity N> must correspond to the "id" of an entity in the "nodes" list.

The Assistant never repeats the same node twice. The Assistant never repeats the same edge twice.
The Assistant responds to the User in JSON only, according to the following JSON schema:

{"type":"object","properties":{"nodes":{"type":"array","items":{"type":"object","properties":{"id":{"type":"string"},"type":{"type":"string"},"detailed_type":{"type":"string"}},"required":["id","type","detailed_type"],"additionalProperties":false}},"edges":{"type":"array","items":{"type":"object","properties":{"from":{"type":"string"},"to":{"type":"string"},"label":{"type":"string"}},"required":["from","to","label"],"additionalProperties":false}}},"required":["nodes","edges"],"additionalProperties":false}
"""
    user_message = f"-------Text begin-------\n{article_text}\n-------Text end-------"
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_message}
    ]
    return messages

def get_nuextract_messages(article_text):
    template = {
        "Nodes": [
            {
                "id": "",
                "type": "",
                "detailed_type": ""
            }
        ],
        "Edges": [
            {
                "from": "",
                "to": "",
                "label": ""
            }
        ]
    }
    template_str = json.dumps(template, indent=4)
    prompt = f"<|input|>\n### Template:\n{template_str}\n### Text:\n{article_text}\n\n<|output|>"
    messages = [
        {"role": "user", "content": prompt}
    ]
    return messages

# --- Benchmark runner function ---
def run_benchmark(model, article_text, extraction_type="default"):
    # Build the message list and options based on extraction type
    if extraction_type == "default":
        messages = get_default_messages(article_text)
        options = {
            "temperature": 0.0,
            "top_p": 0.6,
            "top_k": 30,
        }
        extra_payload = {"keep_alive": "10m"}
    elif extraction_type == "nuextract":
        messages = get_nuextract_messages(article_text)
        options = {"temperature": 0.0}
        extra_payload = {}
    else:
        raise ValueError("Unknown extraction type. Use 'default' or 'nuextract'.")

    num_tokens = calculate_num_tokens(messages)
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    payload.update(extra_payload)

    url = "http://localhost:11434/api/chat"
    print(f"Benchmarking model: {model} | Tokens: {num_tokens}")
    start_time = time.time()

    try:
        response = requests.post(url, json=payload)
        elapsed = time.time() - start_time
        raw_output = response.content.decode('utf-8')
        try:
            parsed_output = json.loads(raw_output)
        except Exception as e:
            parsed_output = f"ERROR: {str(e)} {raw_output}"
    except Exception as e:
        elapsed = None
        raw_output = f"ERROR: {str(e)}"
        parsed_output = raw_output

    result = {
        "model": model,
        "extraction_type": extraction_type,
        "num_tokens": num_tokens,
        "processing_time": elapsed,
        "response_status": response.status_code if 'response' in locals() and response is not None else None,
        "output": parsed_output
    }
    return result

# --- Main benchmark loop ---
def main():
    stop_running_models()

    # Define test texts – add as many as needed
    texts = {
        "Article_1": "Not long after buying and publicly consuming a $6.2m banana as part of an artworld stunt, Chinese crypto entrepreneur Justin Sun made another eye-catching purchase, investing $30m ($23.5m) into a cryptocurrency firm called World Liberty Financial. The company had foundered since its October launch, investors seemingly leery of its prospects and its terms. But it boasted a potentially enticing feature: the chance to do business with a firm partnering with and promoted by none other than Donald Trump. ...",
        "Article_2": "Norway's Jakob Ingebrigtsen cruised to victory at the European Cross Country Championships in Turkey to claim the men's senior title for the third time in four years. The 24-year-old Olympic 5,000m champion chose not to compete in the event last year but reclaimed his crown with a dominant performance at Dokuma Park in Antalya. ...",
        "Article_3": "As Russia continues its aerial bombardment of Ukraine with drones and missiles, Ukraine has been successfully targeting the sources of some of those attacks. One of those was at Engels-2 Airbase, deep inside Russia and which is a key base for Moscow's strategic bombers and also serves as a refuelling point. ...",
        "Article_4": "With her award-winning Wolf Hall series of books, Hilary Mantel made Tudor bad guy Thomas Cromwell sympathetic. But as TV adaptation Wolf Hall: The Mirror and the Light premieres in the US, the question is: did she also 'sidestep crucial matters'? Nearly 500 years after his death, Thomas Cromwell lives again, reborn in the popular imagination ...",
        "Article_5": "The Cook Islands is proving that sustainable tourism isn't just possible – it's essential. Here's how this South Pacific nation is preserving their paradise for generations for come. Landing on Rarotonga, the largest of the Cook Islands chain felt like stepping back in time. ..."
    }

    output_dir = "benchmark_data"
    os.makedirs(output_dir, exist_ok=True)

    # Save the input articles to a JSONL file
    input_filename = os.join(output_dir, "input_articles.jsonl")
    with open(input_filename, "w", encoding="utf-8") as f:
        for text_id, text in texts.items():
            json_obj = {"text_id": text_id, "text": text}
            f.write(json.dumps(json_obj) + "\n")
    print(f"Saved input articles to {input_filename}")

    benchmarks = []
    # Define models to test
    default_models = [
        "hf.co/jackboyla/Phi-3-mini-4k-instruct-graph-GGUF:Q8_0",
        "granite3.1-moe:1b-instruct-fp16",
        "granite3.1-moe:3b-instruct-q8_0"
    ]
    nuextract_models = [
        "hf.co/MaziyarPanahi/NuExtract-1.5-smol-GGUF:Q6_K"
    ]

    # Run benchmarks for each text and model
    for model in default_models:
        for text_id, text in texts.items():
            result = run_benchmark(model, text, extraction_type="default")
            result["text_id"] = text_id
            benchmarks.append(result)
        stop_running_models()
    for model in nuextract_models:
        for text_id, text in texts.items():
            result = run_benchmark(model, text, extraction_type="nuextract")
            result["text_id"] = text_id
            benchmarks.append(result)
        stop_running_models()

    # Save outputs for each model to separate JSONL files
    outputs_by_model = {}
    for res in benchmarks:
        model = res["model"]
        outputs_by_model.setdefault(model, []).append(res)

    for model, results in outputs_by_model.items():
        safe_model_name = model.replace("/", "_").replace(":", "_")
        output_filename = os.path.join(output_dir, f"outputs_{safe_model_name}.jsonl")
        with open(output_filename, "w", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res) + "\n")
        print(f"Saved outputs for model {model} to {output_filename}")

    # Print benchmark results
    print("\nBenchmark Results:")
    for res in benchmarks:
        proc_time = res['processing_time']
        proc_time_str = f"{proc_time:.2f}s" if proc_time is not None else "N/A"
        print(f"Model: {res['model']} | Type: {res['extraction_type']} | {res['text_id']} | Tokens: {res['num_tokens']} | Time: {proc_time_str} | Status: {res['response_status']}")

    # --- Plot processing time vs. token count ---
    markers = {"default": "o", "nuextract": "s"}
    # Automatically generate colors for each unique model
    unique_models = sorted({res["model"] for res in benchmarks})
    colormap = plt.cm.get_cmap("tab20", len(unique_models))
    colors = {model: colormap(i) for i, model in enumerate(unique_models)}

    plt.figure(figsize=(8, 6))
    for res in benchmarks:
        plt.scatter(
            res["num_tokens"], res["processing_time"],
            marker=markers[res["extraction_type"]],
            color=colors.get(res["model"], "black"),
            s=100,
            label=f"{res['model']} ({res['extraction_type']})"
        )
        plt.text(res["num_tokens"]+1, res["processing_time"]+0.1, res["text_id"], fontsize=8)
    plt.xlabel("Approx. Prompt Token Count")
    plt.ylabel("Processing Time (seconds)")
    plt.title("Processing Time vs. Number of Prompt Tokens")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("figs", exist_ok=True)
    plt.savefig("figs/benchmark_results.png")
    plt.show()

if __name__ == "__main__":
    main()
