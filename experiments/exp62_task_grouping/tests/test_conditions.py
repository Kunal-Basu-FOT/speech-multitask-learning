import unittest
from speech_mtl.exp62.conditions import CONDITION_ORDER, get_condition


class ConditionTests(unittest.TestCase):
    def test_frozen_grid(self):
        self.assertEqual(CONDITION_ORDER, ("lid_only", "dialect_only", "lid_dialect"))
        self.assertEqual(get_condition("lid_only").active_tasks, ("lid",))
        self.assertEqual(get_condition("dialect_only").active_tasks, ("dialect",))
        self.assertEqual(get_condition("lid_dialect").active_tasks, ("lid", "dialect"))


if __name__ == "__main__":
    unittest.main()
