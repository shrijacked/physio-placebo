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

That run went through the Hugging Face pipeline. The pipeline's tokenizer
limit is 512, and these prompts are longer than that, so a later check called
`Inferer.__call__` directly: same prompt function, no truncation, 128 new
tokens, then split on `### Response:`. The checkpoint can hold 2048 tokens.
The eight prompts were 545 to 864 tokens, so the question was still in the
window. `pad_token_id` was set to 0, which is what their tokenizer setup does.
Leaving the checkpoint's `-1` makes the current transformers build return
empty strings; that empty run is not a result.

Three fillings of their template, same eight items, same first-number rule:

| How the template was filled | Eight-item MAE | What it wrote |
|---|---|---|
| The dataset's own instruction and the sensor text | 9.86 | Six copies of the sensor list, two refusals |
| The released zero-shot question in the input slot | 6.25 | Five copies, three real lines (`The predicted stress is 3 out of 5` or `Answer: 0`) |
| Health-assistant line as the instruction, sensor text plus the example as the input | 10.0 | Seven copies, one `The predicted stress is 3 out of 5` |

None of these is 0.76. The copies make the first number a step count. Scoring
only the sentences that look like a rating would be a different rule than the
locked one, and it would throw away most of the items. A full 299-item rerun
of this generator would not land on the paper's number.

The locked 2.689 number is still the raw-string run. It has not been replaced.

## The raw question, with nothing cut off

The locked run went through a pipeline whose tokenizer stops at 512 tokens.
Seed 0's 299 questions are 176 to 1133 tokens, median 674. 254 of them are
over 512. The paper says its prompts are 202 ± 42 tokens. The released
questions are longer than that.

The 45 questions that fit in 512 tokens already show the question, including
the `Answer:` example. Their locked answers are still `out of 5` (32),
`out of 10` (5), or `/5` (5).

Generating those strings again with no cut at 512, greedy, 120 new tokens,
reproduces the locked text. Four questions that fit and four that do not all
came back `out of 5` or `/5`, the same words as the locked file. On the
longest questions, a 512-token cut does drop the question (the tail ends in
the middle of the heart-rate list). Putting the question back does not change
what the model writes.

The prompt ends in `Answer: 3` and already contains `[Mood]: 4 out of 5`.
The model continues that phrase. The first number in the continuation is 5
or 10. That is the 2.689.

## Their averaged paragraph, on all 299

Table 1's basic prompt replaces the day-by-day list with the 14-day mean.
Figure 7 puts that paragraph under `Instruction` / `Question` / `Response`.
Figure 8's greedy setting is one beam, no sampling, 128 new tokens. Same
first-number rule. All 299 answers parse.

The model writes `The predicted stress score/level is 2` on 268 items, `3`
on 26, and `1` on 5. MAE is 0.970. Answering 2 every time would be 1.020.
Answering 3 every time would be 0.401. The paper's 0.76 sits between those.
This run does not.

The 31 answers that are not 2 are not random. Mean steps on those questions
are about 5200, against about 3200 when the answer is 2. The five answers of
1 all have mood 5. The model is glancing at steps and mood, then still
saying 2 unless steps are very high. Answering 3 every time is 0.401, which
is better than 0.76, so the paper's number is not "use the steps more
carefully." It is a weaker prediction than always saying 3.

## The other wordings in the paper, eight items

Figure 7's bullet list of the days, with "can you predict user's stress level",
makes the model refuse. All eight answers are "I cannot predict stress levels".
Rounding heart rate to two decimals, as the figure prints `65.61`, does not
change that. There is no number to score.

Table 2 ends the stress prompt with "What will my stress level be?" On the
same averages, seven answers are "Your stress level will be 2 out of 5." One
is "Your stress level will be low."

Figure 8's sampling setting (temperature 0.4, top 50, top-p 0.9) on the
averaged paragraph gives six 2s and two 3s.

Figure 3's sentence, "Please return the stress level between 1 and 5", on
the same averages: seven 3s and one 4. The eight-item error is 0.5. Answering
3 on all 299 questions is 0.401. The same sentence on Figure 3's bracketed
day list, heart rate rounded to one decimal, answers `3` on all eight.

Figure 8 sampling at temperature 0.4, on the released raw question, still
returns `out of 5` on all eight. The first number is 5. Temperature 0.7,
their other sampling block, does the same: eight answers, first number 5
(`out of 5`, `/5`, or `/5, low stress`).

Their few-shot branch, three averaged examples plus the same `Answer: {n}`
line, fits in 520 tokens and then generates nothing. All eight completions
are empty. The prompt already ends in `Answer: 3`, and the examples have
already shown a finished answer, so the model stops.

None of these moves the full-set 0.970 result onto 0.76.

The instruction stored on each stress row ("a personalized healthcare agent
trained to predict stress which ranges from 1 to 5") with the same averages
makes all eight answers "the predicted stress level is 3." Answering 3 on
every one of the 299 questions scores 0.401. That is not 0.76 either.

The locked 2.689 number is still the raw-string run. It has not been replaced.

## Their repository has nothing else

`mitmedialab/Health-LLM` at main (`ca095854`) is the same tree we have:
`inference.py`, `gen_dataset.py`, the MedAlpaca inferer, and the paper PDF.
There is no evaluation script and no other prompt file. The 0.76 number is
not recoverable from those files with this checkpoint and the first-number
score.
