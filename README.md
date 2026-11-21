# Multi-hop question answering with reliability-weighted reasoning chains

A question can have a well-supported answer even when several attempts to solve it disagree. This architecture generates candidate answers in two ways: a reader answers directly from retrieved passages, and a planner breaks the question into hops that the reader solves in sequence. Two additional chains can draw on a memory of successful plans. Each chain leaves an evidence trace that a small model uses to estimate how much support to give its answer.

![Figure 2. On a HotpotQA question, a two-hop chain quotes evidence for Gabriella Montez. Two memory-guided chains and the direct reader support that answer, giving it a pooled support of 0.998.](docs/images/fig2-overview.png)

## Following a chain of evidence

For a question such as “In which country was the director of *Spirited Away* born?”, the planner can produce two hops: “Who directed *Spirited Away*?” and “In which country was #1 born?” The retriever finds passages for the first hop. The model extracts an intermediate answer and a supporting quotation, then substitutes that answer for `#1` before the second retrieval. A final reader receives the original question, selected passages, and notes from the chain.

NV-Embed-v2 embeds the retrieval queries. A shared Qwen3.5-9B model plans the hops, extracts intermediate answers, and reads the evidence. Both base models have fixed weights. The direct path skips planning and reads the five passages retrieved for the original question. [Chain execution and direct reader](src/hyperbind/chain_agent.py)

## Executions used by SupportChain

The full method in the main results table merges the four executions below by calibrated support.

| Path | How it constructs the answer |
| --- | --- |
| Direct reader (`a0`) | Read passages retrieved with the original question. |
| Fixed-example chain (`single`) | Plan from a fixed set of examples, then retrieve and answer one hop at a time. |
| First memory-guided chain (`selfmem_answer`) | Retrieve similar training questions by embedding similarity and use their successful plans as examples. Execute the new plan against the passage corpus. |
| Second memory-guided chain (`selfmem2_answer`) | Use a memory expanded with successful plans from the first memory-guided round. Execute the resulting plan with the same retriever and reader. |

Planning memory stores examples of **how to decompose a question**. During construction, the answer-based admission rule keeps completed training chains with valid plans whose answers meet an F1 threshold (0.8 by default). At inference, the planner receives the retrieved questions and their hop sequences. [Memory construction](scripts/build_plan_memory.py) · [Four-path configuration](configs/final.json)

## Ablations and comparisons

| Experiment | What changes |
| --- | --- |
| Execution ablation | Keep support merging and remove executions. For example, “Single + 2 plans” omits the dense read from the full method. |
| Execution-source comparison | Replace the two memory-guided executions with two sampled executions of the fixed-example chain. The appendix reports this comparison both with and without the dense read. |
| Merge-rule comparison | Give each rule the same execution outputs. Compare calibrated support with majority vote, the most reliable execution, and the reader choosing an answer. |
| Inference-cost comparison | The full method runs all four executions. The on-demand variant runs the two memory-guided executions only when the dense read and the fixed-example chain disagree. |

The configuration file records when the four-execution reference was selected for reporting. [Configuration](configs/final.json) · [Evaluation](scripts/paper_results.py) · [Paper tables](scripts/paper_data.py)

## Selecting the final answer

A logistic regression assigns each chain a reliability score from its execution trace. Its inputs include whether the chain finished, whether the chain and the final reader agreed, whether a quotation appeared verbatim in a passage, and how highly the evidence ranked. The direct reader receives a dataset-specific prior. The fitted coefficients and normalization parameters are included in the repository. [Reliability model](src/hyperbind/support.py) · [Fitted parameters](configs/support)

The selector groups matching answers. For a group with reliability scores *p₁, …, pₙ*, its support is **1 − ∏ᵢ(1 − pᵢ)**. For example, two paths supporting the same answer with scores 0.6 and 0.5 give it a group score of 0.8. If another answer has a single supporting path scored at 0.7, the first answer wins. The selector returns the answer text from the most reliable path in the winning group. [Answer matching](src/hyperbind/textmatch.py) · [Grouping and selection](src/hyperbind/support.py)

## Running a single chain

From a prepared Linux workspace with the dataset split, passage embeddings, and model services available, run:

```bash
PYTHONPATH=src envs/main/bin/python scripts/run_chain.py \
  --dataset musique --split test --tag single --candidates 1
```

This command exercises the fixed-example chain. The runner records the plan, hop evidence, intermediate answers, and final reader output. [Chain runner](scripts/run_chain.py) · [Environment and data preparation](preparation)
