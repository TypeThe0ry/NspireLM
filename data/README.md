# data

`tinystories_usage.npy` holds a uint32 count for each of the 32000 llama2 tokenizer ids. The counts come
from the first 300 MB of `TinyStories-train.txt` (85.4M tokens; huggingface.co/datasets/roneneldan/TinyStories),
tokenized with llama2.c's `tokenizer.model`. To regenerate:

    python3 tools/vocab_usage.py tokenizer.model TinyStories-train.txt data/tinystories_usage.npy --max-mb 300

`nslmc --vocab-usage` uses the file to compile out tokens that never occur in the training corpus.
Of the 32000 tokens, 12750 remain (including every byte token and BOS/EOS). This removes 60% of the
stories15M classifier at a measured perplexity cost of under 0.01%.
