import unittest
from custom_types import *
from prosumer import *


class ProsumerTest(unittest.TestCase):
    def test_flexibility_reward(self):
        fixed = lambda job, t: 1 if job[1] == t else 0

        c = 0.4
        linear = lambda job, t: max(1 - abs(job[1] - t) * c, 0)

        free = lambda job, t: 1

        fixed_job = (4.5, 4, fixed)

        self.assertEqual(fixed(fixed_job, 4), 1)
        self.assertEqual(fixed_job[2](fixed_job, 4), 1)
        self.assertEqual(fixed_job[2](fixed_job, 5), 0)

        free_job = (3.1, 3, free)
        self.assertEqual(free_job[2](free_job, 5), 1)
        self.assertEqual(free_job[2](free_job, 3), 1)
        self.assertEqual(free_job[2](free_job, 0), 1)

        linear_job = (3, 2, linear)
        self.assertEqual(linear_job[2](linear_job, 2), 1)
        self.assertEqual(linear_job[2](linear_job, 1), 0.6)
        self.assertAlmostEqual(linear_job[2](linear_job, 4), 0.2)
        self.assertEqual(linear_job[2](linear_job, 10), 0)

    def test_constructor(self):
        fixed = lambda job, t: 1 if job[1] == t else 0

        a = 0.4
        linear = lambda job, t: max(1 - abs(job[1] - t) * a, 0)

        free = lambda job, t: 1

        prosumer = ProsumerAgent(
            1,
            [(4.5, 4, fixed), (3.1, 3, free), (3, 14, linear)],
            24,
            5,
            generation_type="solar",
        )

        self.assertIsInstance(prosumer, ProsumerAgent)
        self.assertEqual(5, prosumer.generation_capacity)
        self.assertEqual(4.1, prosumer.schedule[3])

        prosumer.calculate_net_demand(3)
        self.assertEqual(4.1, prosumer.net_demand)
        prosumer.calculate_net_demand(14)
        self.assertGreater(4, prosumer.net_demand)


if __name__ == "__main__":
    unittest.main()
