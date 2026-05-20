
mkdir -p output
mkdir -p shared
hf download OpenRLHF/Llama-3-8b-sft-mixture --local-dir ./shared/Llama-3-8b-sft-mixture
hf download OpenRLHF/Llama-3-8b-rm-700k --local-dir ./shared/Llama-3-8b-rm-700k
hf download OpenRLHF/prompt-collection-v0.1 --repo-type=dataset --local-dir ./shared/prompt-collection-v0.1
pip install -e .
