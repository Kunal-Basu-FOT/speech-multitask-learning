import unittest
from speech_mtl.exp62.data import StableFrameBudgetBatchSampler


class BatchingTests(unittest.TestCase):
    def test_budget_and_stable_count(self):
        lengths = [47, 51, 70, 100, 200, 500, 13000]
        sampler = StableFrameBudgetBatchSampler(lengths, 1000, 4, 3, 1337, True)
        count = len(sampler)
        for epoch in (1, 2, 40):
            sampler.set_epoch(epoch)
            batches = list(sampler)
            self.assertEqual(len(batches), count)
            flattened = sorted(i for batch in batches for i in batch)
            self.assertEqual(flattened, list(range(len(lengths))))
            for batch in batches:
                cost = max(lengths[i] for i in batch) * len(batch)
                self.assertTrue(cost <= 1000 or len(batch) == 1)


if __name__ == "__main__":
    unittest.main()
