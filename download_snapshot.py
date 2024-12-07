from huggingface_hub import snapshot_download
import os
os.environ['HF_TOKEN'] = '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'
model_id="EmergentMethods/Phi-3-mini-4k-instruct-graph"
snapshot_download(repo_id=model_id, local_dir="Phi-3-mini-4k-instruct-graph",
                  local_dir_use_symlinks=False, revision="main")