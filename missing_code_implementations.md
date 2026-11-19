# Missing Code Implementations — Train a Small Language Model

This document collects the missing Python code sections for the **Google DeepMind: Train a Small Language Model** challenge lab on Google Skills.

The challenge uses a dataset of Arabic children's stories and covers three foundational tasks:

1. Generating text with an N-gram language model.
2. Splitting token sequences into fixed-length, overlapping segments.
3. Encoding and preparing dataset text for a character-based Transformer language model.

> **Note:** These are code implementations for the sections described below. They assume the notebook already defines the `tokenizer`, `ngram_model`, `generated_tokens`, `dataset`, and the relevant configuration variables.

---

## 1. N-gram text generation

Use the following code to continue generating tokens from an N-gram model. The model looks up the most recent context, selects the next token using the requested sampling method, and appends that token to the generated sequence.

```python
if not ngram_model:
    return tokenizer.join_text(generated_tokens)

context_size = len(
    tokenizer.character_tokenize(next(iter(ngram_model)))
)

for _ in range(n_tokens):
    if len(generated_tokens) < context_size:
        break

    context_tokens = generated_tokens[-context_size:]
    context_key = tokenizer.join_text(context_tokens)

    if context_key not in ngram_model:
        break

    next_token_distribution = ngram_model[context_key]

    if not next_token_distribution:
        break

    if sampling_mode == "random":
        tokens = list(next_token_distribution.keys())
        probabilities = list(next_token_distribution.values())

        next_token = random.choices(
            tokens,
            weights=probabilities,
            k=1
        )[0]

    elif sampling_mode == "greedy":
        next_token = max(
            next_token_distribution,
            key=next_token_distribution.get
        )

    else:
        raise ValueError(
            f"Unsupported sampling_mode: '{sampling_mode}'"
        )

    generated_tokens.append(next_token)

return tokenizer.join_text(generated_tokens)
```

### How it works

- **Empty model:** Returns the text generated so far if the N-gram model is empty.
- **Context size:** Determines how many recent characters/tokens are needed to look up the next-token distribution.
- **Context lookup:** Uses the latest generated tokens as the key for finding possible next tokens.
- **Random sampling:** Selects a next token using the weights in the model's distribution.
- **Greedy sampling:** Selects the most likely next token.
- **Stopping conditions:** Stops when the context is too short, the context is not present in the model, or no next-token options are available.

**Sampling modes:** Use `"random"` or `"greedy"`. The spelling `"greddy"` in the original snippet has been corrected to `"greedy"`; make sure the value passed to `sampling_mode` uses the corrected spelling too.

The snippet uses `random.choices`, so the notebook or Python file must include:

```python
import random
```

---

## 2. Segmenting a token sequence

The following function divides an encoded sequence into smaller segments. It supports overlap between consecutive segments so that neighbouring training examples can share context.

```python
def segment_encoded_sequence(sequence, max_length, n_overlap):
    if max_length <= 0:
        raise ValueError("max_length must be greater than 0")

    if n_overlap < 0 or n_overlap >= max_length:
        raise ValueError("n_overlap must be at least 0 and less than max_length")

    subsequences = []
    start = 0

    while start < len(sequence):
        end = start + max_length
        subsequences.append(sequence[start:end])

        if end >= len(sequence):
            break

        start = end - n_overlap

    return subsequences
```

### How it works

- `max_length` controls the maximum number of token IDs in each segment.
- `n_overlap` controls how many token IDs are shared between consecutive segments.
- The loop advances by `max_length - n_overlap` until the sequence has been covered.
- The final segment may be shorter than `max_length`.

**Example:** With `max_length=8` and `n_overlap=2`, each new segment starts 6 positions after the previous segment's start.

---

## 3. Encode and segment the dataset

Use this code to tokenize every text sample in the dataset, split its token IDs into segments, and collect the resulting segments for model training.

```python
encoded_tokens = []

for text in dataset:
    token_ids = tokenizer.encode(text)

    segments = segment_encoded_sequence(
        token_ids,
        segmentation_length,
        n_overlap
    )

    encoded_tokens.extend(segments)
```

### How it works

1. Iterates through each text sample in `dataset`.
2. Converts the text into token IDs with `tokenizer.encode(text)`.
3. Splits the encoded sequence using `segment_encoded_sequence`.
4. Adds all generated segments to `encoded_tokens`.

After this step, `encoded_tokens` contains a list of token-ID segments that can be used in the subsequent data preparation and training workflow.

---

## Important implementation notes

- Ensure the tokenizer's `encode`, `character_tokenize`, and `join_text` methods are available and behave consistently with the notebook's token representation.
- Ensure `ngram_model` maps each context string to a non-empty dictionary of possible next tokens and their sampling weights.
- For random sampling, the values in each next-token distribution should be valid non-negative weights, with at least one positive weight.
- Ensure `segmentation_length` is greater than zero and `n_overlap` is smaller than `segmentation_length`.
- If the surrounding notebook uses a different sampling-mode label, update both the function and its caller consistently.

## Learning topics

- Character-level tokenization
- N-gram language modelling
- Random and greedy text generation
- Token sequence segmentation
- Dataset preparation for Transformer models
- Foundations of small language models

## Learning resource

[Google DeepMind AI Research Foundations — Google Skills](https://www.skills.google/paths/3135)

[Build Your Own Small Language Model](https://goo.gle/build-small-LM)
