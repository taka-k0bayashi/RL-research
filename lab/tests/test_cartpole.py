import unittest

import numpy as np

from rl_lab.algorithms import discounted_returns, evaluate, train
from rl_lab.envs import make_environment
from rl_lab.models import make_model
from rl_lab.plot_results import moving_average


class DiscountedReturnsTest(unittest.TestCase):
    def test_discounted_returns(self) -> None:
        self.assertEqual(
            discounted_returns([1.0, 1.0, 1.0], 0.5).tolist(),
            [1.75, 1.5, 1.0],
        )

    def test_model_sizes(self) -> None:
        expected = {"linear": 10, "mlp_32": 226, "mlp_64x64": 4610}
        for name, parameters in expected.items():
            with self.subTest(model=name):
                model = make_model(name, 4, 2)
                self.assertEqual(
                    sum(parameter.numel() for parameter in model.parameters()),
                    parameters,
                )

    def test_cartpole_adapter(self) -> None:
        environment = make_environment("cartpole")
        try:
            self.assertEqual(environment.observation_size, 4)
            self.assertEqual(environment.action_size, 2)
            observation = environment.reset(seed=0)
            next_observation, reward, done = environment.step(0)
            self.assertEqual(observation.shape, (4,))
            self.assertEqual(next_observation.shape, (4,))
            self.assertEqual(reward, 1.0)
            self.assertIsInstance(done, bool)
        finally:
            environment.close()

    def test_reinforce_smoke(self) -> None:
        model = make_model("linear", 4, 2)
        scores = train("cartpole", model, 0, 1, 0.99, 0.01, "smoke")
        evaluation = evaluate("cartpole", model, 0, 1)
        self.assertEqual(len(scores), 1)
        self.assertEqual(len(evaluation), 1)

    def test_moving_average(self) -> None:
        np.testing.assert_allclose(
            moving_average(np.array([1.0, 2.0, 3.0, 4.0]), 2),
            [1.5, 2.5, 3.5],
        )


if __name__ == "__main__":
    unittest.main()
