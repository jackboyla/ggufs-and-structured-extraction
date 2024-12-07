1. Run download_snapshot.py for your desired model
2. Clone llama.cpp. Ideally use the latest version, but if there is no `convert_hf_to_gguf.py` file, you can run `git checkout 19d8762`.
3. `pip install -r llama.cpp/requirements.txt`
4. Check that your model has a `tokenizer.model` file. If not, you'll need to get it from the base model. e.g for `Phi-3-mini-4k-instruct-graph`, there was no such file, so I downloaded it from the original [Phi-3-mini repo](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/tree/main). Put this file in your downloaded model's dir. PLEASE NOTE: if the tokenizer/vocab was modified from the base model to your desried finetuned model, this approach will likely cause issues.
5. Run:
```bash
python llama.cpp/convert_hf_to_gguf.py create-gguf/Phi-3-mini-4k-instruct-graph \
  --outfile create-gguf/Phi-3-mini-4k-instruct-graph.Q8_0.gguf \
  --outtype q8_0
```

To add to a huggingface model repo, follow [these steps](https://huggingface.co/docs/hub/en/repositories-getting-started#terminal)

To clear huggingface cache:
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli delete-cache
```
