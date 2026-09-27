import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from rl_lab.algorithms import (
    discounted_returns,
    evaluate,
    train_actor_critic,
    train_reinforce,
)
from rl_lab.algorithms.distillation import train as train_distillation
from rl_lab.cartpole import load_config, save_run
from rl_lab.envs import make_environment
from rl_lab.models import make_model
from rl_lab.plot_results import moving_average


class CartPoleTest(unittest.TestCase):
    def test_discounted_returns(self) -> None:
        self.assertEqual(
            discounted_returns([1.0, 1.0, 1.0], 0.5).tolist(),
            [1.75, 1.5, 1.0],
        )

    def test_model_sizes(self) -> None:
        expected = {"linear": 15, "mlp_32": 259, "mlp_64x64": 4675}
        for name, parameters in expected.items():
            with self.subTest(model=name):
                model = make_model(name, (4,), 2)
                self.assertEqual(
                    sum(parameter.numel() for parameter in model.parameters()),
                    parameters,
                )

    def test_cartpole_adapter(self) -> None:
        environment = make_environment("cartpole")
        try:
            self.assertEqual(environment.observation_shape, (4,))
            self.assertEqual(environment.action_size, 2)
            observation = environment.reset(seed=0)
            next_observation, reward, done = environment.step(0)
            self.assertEqual(observation.shape, (4,))
            self.assertEqual(next_observation.shape, (4,))
            self.assertEqual(reward, 1.0)
            self.assertIsInstance(done, bool)
        finally:
            environment.close()

    def test_actor_critic_smoke(self) -> None:
        model = make_model("linear", (4,), 2)
        scores, interrupted, _ = train_actor_critic(
            "cartpole",
            model,
            0,
            1,
            0.999,
            0.95,
            0.01,
            0.01,
            0.0,
            0.5,
            1.0,
            1,
            "smoke",
        )
        evaluation = evaluate("cartpole", model, 0, 2, 2)
        self.assertEqual(len(scores), 1)
        self.assertFalse(interrupted)
        self.assertEqual(len(evaluation), 2)

    def test_reinforce_selection(self) -> None:
        config = load_config(Path("experiments/cartpole_reinforce.toml"))
        model = make_model("linear", (4,), 2)
        model.critic.requires_grad_(False)
        scores, interrupted, _ = train_reinforce(
            "cartpole", model, 0, 1, 0.99, 0.01, "smoke"
        )
        self.assertEqual(config.algorithm, "reinforce")
        self.assertEqual(len(scores), 1)
        self.assertFalse(interrupted)
        self.assertEqual(
            sum(p.numel() for p in model.parameters() if p.requires_grad), 10
        )

    def test_training_interrupt(self) -> None:
        class OneStepEnvironment:
            def reset(self, seed: int) -> np.ndarray:
                return np.zeros(4, dtype=np.float32)

            def step(self, action: int) -> tuple[np.ndarray, float, bool]:
                return np.zeros(4, dtype=np.float32), 1.0, True

            def close(self) -> None:
                pass

        class InterruptingEnvironment:
            def __init__(self) -> None:
                self.episodes = 0

            def reset(self, seed: int) -> np.ndarray:
                if self.episodes == 1:
                    raise KeyboardInterrupt
                self.episodes += 1
                return np.zeros(4, dtype=np.float32)

            def step(self, action: int) -> tuple[np.ndarray, float, bool]:
                return np.zeros(4, dtype=np.float32), 1.0, True

            def close(self) -> None:
                pass

        model = make_model("linear", (4,), 2)
        with patch(
            "rl_lab.algorithms.actor_critic.make_environment",
            return_value=InterruptingEnvironment(),
        ):
            scores, interrupted, checkpoint_state = train_actor_critic(
                "interrupt",
                model,
                0,
                2,
                0.999,
                0.95,
                0.01,
                0.01,
                0.0,
                0.5,
                1.0,
                1,
                "interrupt",
            )
        self.assertEqual(scores, [1.0])
        self.assertTrue(interrupted)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            save_run(
                output,
                "actor_critic",
                "linear",
                0,
                model,
                scores,
                [],
                interrupted,
                checkpoint_state,
            )
            summary = json.loads((output / "summary.json").read_text())
            self.assertTrue(summary["interrupted"])
            self.assertIsNone(summary["mean_evaluation_return"])
            self.assertTrue((output / "model.pt").is_file())
            checkpoint = torch.load(
                output / "checkpoint.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(checkpoint["episode"], 1)

            resumed_model = make_model("linear", (4,), 2)
            resumed_model.load_state_dict(checkpoint["model_state_dict"])
            with patch(
                "rl_lab.algorithms.actor_critic.make_environment",
                return_value=OneStepEnvironment(),
            ):
                resumed_scores, resumed_interrupted, _ = train_actor_critic(
                    "resume",
                    resumed_model,
                    0,
                    2,
                    0.999,
                    0.95,
                    0.01,
                    0.01,
                    0.0,
                    0.5,
                    1.0,
                    1,
                    "resume",
                    start_episode=checkpoint["episode"],
                    initial_scores=checkpoint["scores"],
                    optimizer_state=checkpoint["optimizer_state_dict"],
                    torch_rng_state=checkpoint["torch_rng_state"],
                    cuda_rng_states=checkpoint["cuda_rng_state_all"],
                )
            self.assertEqual(resumed_scores, [1.0, 1.0])
            self.assertFalse(resumed_interrupted)

    def test_pixel_cartpole_with_cnn(self) -> None:
        environment = make_environment("cartpole_pixels")
        try:
            observation = environment.reset(seed=0)
            next_observation, reward, done = environment.step(0)
            self.assertEqual(observation.shape, (4, 84, 84))
            self.assertEqual(next_observation.shape, (4, 84, 84))
            self.assertEqual(environment.teacher_observation().shape, (4,))
            self.assertEqual(observation.dtype, np.float32)
            self.assertGreaterEqual(float(observation.min()), -1.0)
            self.assertLessEqual(float(observation.max()), 1.0)
            self.assertLess(float(observation.mean()), 0.1)
            np.testing.assert_array_equal(observation[1:], 0.0)
            self.assertEqual(reward, 1.0)
            self.assertIsInstance(done, bool)
        finally:
            environment.close()

        expected = {
            "cnn_16x32_fc128": 344_627,
            "cnn_16x32_fc512x128": 1_406_003,
        }
        for name, parameters in expected.items():
            with self.subTest(model=name):
                model = make_model(name, (4, 84, 84), 2)
                self.assertEqual(model(torch.from_numpy(observation)).shape, (2,))
                logits, value = model.actor_critic(torch.from_numpy(observation))
                self.assertEqual(logits.shape, (2,))
                self.assertEqual(value.shape, ())
                self.assertEqual(
                    sum(parameter.numel() for parameter in model.parameters()),
                    parameters,
                )

    def test_distillation_smoke(self) -> None:
        teacher = make_model("mlp_32", (4,), 2)
        student = make_model("cnn_16x32_fc128", (4, 84, 84), 2)
        student.critic.requires_grad_(False)
        scores, losses, interrupted = train_distillation(
            teacher, student, 0, 1, 0.0003, 1
        )
        self.assertEqual(len(scores), 1)
        self.assertTrue(losses)
        self.assertFalse(interrupted)

    def test_moving_average(self) -> None:
        np.testing.assert_allclose(
            moving_average(np.array([1.0, 2.0, 3.0, 4.0]), 2),
            [1.5, 2.5, 3.5],
        )


if __name__ == "__main__":
    unittest.main()
