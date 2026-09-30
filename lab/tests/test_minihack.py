import sys
import unittest

import numpy as np

from rl_lab.envs import make_environment


@unittest.skipIf(sys.platform == "win32", "MiniHack requires Linux or macOS")
class MiniHackTest(unittest.TestCase):
    def test_room_adapter(self) -> None:
        environment = make_environment("minihack_room_5x5")
        try:
            observation = environment.reset(seed=0)
            next_observation, reward, done = environment.step(0)
            self.assertEqual(environment.observation_shape, (81,))
            self.assertEqual(environment.action_size, 8)
            self.assertEqual(observation.shape, (81,))
            self.assertEqual(next_observation.dtype, np.float32)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(done, bool)
        finally:
            environment.close()


if __name__ == "__main__":
    unittest.main()
