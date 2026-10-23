# Why the prompting-only rerun is 2.69, not 0.76

Health-LLM Table 3 says zero-shot MedAlpaca-7B on PMData stress has MAE 0.76 ± 0.1.
The locked rerun is 2.689 ± 0.038 (seeds 2.714, 2.645, 2.708). The prompts are the
released `inference.py` zero-shot string. The model is `medalpaca/medalpaca-7b`.
They shipped no evaluation code, so the score is the first number in the completion.

## What the model actually wrote

On seed 0, 218 of 299 completions are exactly `out of 5`. Another 42 are `out of 10`.
The first number in those strings is 5 or 10. Stress labels only go to 5, so the
error is large. The same pattern appears on seeds 1 and 2. This is not one unlucky seed.

The prompt already ends with an example, `Answer: 3`, and the input contains
`[Mood]: 4 out of 5`. MedAlpaca continues that phrase. It does not emit a new
stress rating.

## Re-reading the same answers does not produce 0.76

| Reading of the saved text | MAE |
|---|---|
| First number in the completion (the locked rule) | 2.71 / 2.65 / 2.71 |
| Last number in the completion | same |
| The `Answer:` number already sitting in the prompt | about 1.6 |
| The mood number already sitting in the prompt | 0.42 |
| Always answer 3 | 0.40 |

None of these is 0.76. The completions do not contain a 1–5 stress prediction to recover.

## What does land near 0.76

Qwen3-8B, given the same prompt text inside its chat template, answers `Answer: 3`.
Its MAE is 0.795 ± 0.020, inside the paper's 0.76 ± 0.1. Always answering 3 is still
better (0.401). So a model that actually answers the question lands on their prompting
number. MedAlpaca, fed the raw string, does not answer the question.

## What is not the same, even though the words of the prompt are

- Their script calls `medalpaca_pl(question)` and never defines that function.
  There is no published generation setting for this cell.
- This rerun is greedy (`do_sample` false). Their Gemini calls use temperature 0.9.
  That setting is not attached to the MedAlpaca branch. Three seeds all producing
  `out of 5` is the model's usual completion, not a sampling fluke.
- MedAlpaca was trained to answer after `### Instruction:` / `### Response:`.
  The released zero-shot string does not use that wrapper. The raw string ends in
  `Answer: 3`, and the model keeps writing.

Changing the locked first-number rule cannot turn these completions into 0.76.

## Their own prompt function, tried

`DataHandler.generate_prompt` in `medalpaca/handler.py` is how MedAlpaca builds a
prompt. The released zero-shot words were passed in as the instruction and the
sensor text, which is the two-argument call that function has. Eight items,
seed 0, same first-number rule:

Seven answers copy the sensor list, so the first number is a step count.
One answer is `The predicted stress is 3 out of 5.` The eight-item MAE is 10,
worse than 2.69. A full rerun of this wrapper would not land on 0.76.

The locked 2.689 number is still the raw-string run. It has not been replaced.
