"""The chat workers' generation batches: without a character budget they are plain BATCH-sized slices (every run so
far)."""



from chameleons.chat import BATCH, batch_prompts # noqa: E402


def test_without_a_budget_the_batches_are_the_plain_slices():
    prompts = ["x" * (k % 7 + 1) for k in range(2 * BATCH + 5)]
    assert batch_prompts(prompts) == [list(range(i, min(i + BATCH, len(prompts))))
                                      for i in range(0, len(prompts), BATCH)]
