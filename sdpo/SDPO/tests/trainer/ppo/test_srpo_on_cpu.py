# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU unit tests for SRPO (Sample-Routed Policy Optimization) loss and routing."""

import unittest

import torch

from verl.trainer.ppo.core_algos import compute_srpo_loss


class _DummyActorConfig:
    """Minimal stand-in for ActorConfig used by compute_srpo_loss."""

    def __init__(self):
        self.clip_ratio = 0.2
        self.clip_ratio_low = 0.2
        self.clip_ratio_high = 0.28

    def get(self, key, default=None):
        return getattr(self, key, default)


class _DummySDConfig:
    def __init__(self, alpha=0.5, is_clip=None, add_tail=True):
        self.alpha = alpha
        self.is_clip = is_clip
        self.distillation_add_tail = add_tail

    def get(self, key, default=None):
        return getattr(self, key, default)


class _DummySRPOConfig:
    def __init__(self, dw_beta=1.0):
        self.dw_beta = dw_beta
        self.dw_normalizer_scope = "global"

    def get(self, key, default=None):
        return getattr(self, key, default)


def _make_topk_logprobs(bsz, seq, k, peaky=False, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(bsz, seq, k)
    if peaky:
        logits = logits * 8.0  # low-entropy (peaky) distribution
    return torch.log_softmax(logits, dim=-1)


class TestSRPOLoss(unittest.TestCase):
    def setUp(self):
        self.bsz, self.seq, self.k = 4, 5, 6
        self.old_log_prob = torch.zeros(self.bsz, self.seq)
        self.log_prob = torch.zeros(self.bsz, self.seq)  # ratio == 1
        self.advantages = torch.randn(self.bsz, self.seq)
        self.response_mask = torch.ones(self.bsz, self.seq)
        self.student_topk = _make_topk_logprobs(self.bsz, self.seq, self.k, seed=1)
        self.teacher_topk = _make_topk_logprobs(self.bsz, self.seq, self.k, seed=2)
        self.cfg = _DummyActorConfig()
        self.sd = _DummySDConfig()
        self.srpo = _DummySRPOConfig()

    def _call(self, srpo_sdpo_mask, srpo_cfg=None, sd_cfg=None):
        return compute_srpo_loss(
            old_log_prob=self.old_log_prob,
            log_prob=self.log_prob,
            advantages=self.advantages,
            response_mask=self.response_mask,
            srpo_sdpo_mask=srpo_sdpo_mask,
            student_topk_log_probs=self.student_topk,
            teacher_topk_log_probs=self.teacher_topk,
            self_distillation_config=sd_cfg or self.sd,
            srpo_config=srpo_cfg or self.srpo,
            config=self.cfg,
        )

    def test_all_grpo_reduces_to_grpo_mean(self):
        """srpo_sdpo_mask all-zero => loss equals GRPO masked_sum / total tokens (ratio=1)."""
        mask = torch.zeros(self.bsz)
        loss, metrics = self._call(mask)
        # ratio == 1, advantages<0 dual-clip inactive at ratio 1 -> per-token = -adv
        expected = (-self.advantages * self.response_mask).sum() / self.response_mask.sum()
        self.assertTrue(torch.allclose(loss, expected, atol=1e-5))
        self.assertEqual(metrics["srpo/sdpo_tokens"], 0.0)
        self.assertAlmostEqual(metrics["srpo/sdpo_frac"], 0.0, places=6)

    def test_joint_denominator(self):
        """Mixed routing uses a single shared denominator = total routed tokens."""
        mask = torch.tensor([0.0, 0.0, 1.0, 1.0])
        loss, metrics = self._call(mask)
        total = self.response_mask.sum().item()
        self.assertAlmostEqual(
            metrics["srpo/grpo_tokens"] + metrics["srpo/sdpo_tokens"], total, places=5
        )
        # sdpo rows are 2 of 4 -> half the tokens
        self.assertAlmostEqual(metrics["srpo/sdpo_frac"], 0.5, places=5)
        self.assertTrue(torch.isfinite(loss))

    def test_dynamic_weighting_beta_zero_is_uniform(self):
        """beta=0 => all dynamic weights equal 1 (mean over sdpo tokens == 1)."""
        mask = torch.tensor([0.0, 1.0, 1.0, 1.0])
        _, metrics = self._call(mask, srpo_cfg=_DummySRPOConfig(dw_beta=0.0))
        self.assertAlmostEqual(metrics["srpo/dw_weight_mean"], 1.0, places=5)

    def test_dynamic_weighting_normalized_mean_is_one(self):
        """For any beta, normalized weights average to 1 over the SDPO token set."""
        mask = torch.tensor([0.0, 1.0, 1.0, 1.0])
        _, metrics = self._call(mask, srpo_cfg=_DummySRPOConfig(dw_beta=1.0))
        self.assertAlmostEqual(metrics["srpo/dw_weight_mean"], 1.0, places=4)

    def test_no_nan_when_no_sdpo_tokens(self):
        mask = torch.zeros(self.bsz)
        loss, _ = self._call(mask)
        self.assertFalse(torch.isnan(loss).any())


class TestSRPORoutingMask(unittest.TestCase):
    """Replicates the srpo_sdpo_mask formula z_i = (1-c_i)*m_i*(1-reroll)."""

    @staticmethod
    def _route(sd_mask, is_wrong, reroll):
        return sd_mask * is_wrong * (1.0 - reroll)

    def test_table7_cases(self):
        # (correct, teacher_avail) -> expected SDPO route (1) or GRPO (0)
        # correct+avail -> GRPO(0); correct+no -> GRPO(0);
        # wrong+avail -> SDPO(1); wrong+no -> GRPO fallback(0)
        sd_mask = torch.tensor([1.0, 0.0, 1.0, 0.0])      # teacher available (m_i)
        is_wrong = torch.tensor([0.0, 0.0, 1.0, 1.0])     # (1-c_i)
        reroll = torch.zeros(4)
        out = self._route(sd_mask, is_wrong, reroll)
        self.assertTrue(torch.equal(out, torch.tensor([0.0, 0.0, 1.0, 0.0])))

    def test_reroll_group_excluded(self):
        sd_mask = torch.tensor([1.0, 1.0])
        is_wrong = torch.tensor([1.0, 1.0])
        reroll = torch.tensor([1.0, 0.0])  # first sample is in a re-rolled group
        out = self._route(sd_mask, is_wrong, reroll)
        self.assertTrue(torch.equal(out, torch.tensor([0.0, 1.0])))


class TestRerollAllocation(unittest.TestCase):
    """Replicates the free/forced slot allocation used in _maybe_build_reroll_batch."""

    @staticmethod
    def _alloc(n, num_forced, rolls_per, distinct):
        forced_slots = min(num_forced, distinct)
        free_rolls = n - forced_slots * rolls_per
        if free_rolls < 0:
            forced_slots = n // rolls_per
            free_rolls = n - forced_slots * rolls_per
        return free_rolls, forced_slots

    def test_group_size_preserved(self):
        n, num_forced, rolls_per = 8, 3, 2
        for distinct in range(0, 6):
            free, forced = self._alloc(n, num_forced, rolls_per, distinct)
            self.assertGreaterEqual(free, 0)
            self.assertEqual(free + forced * rolls_per, n)

    def test_normal_case_two_free_six_forced(self):
        free, forced = self._alloc(8, 3, 2, distinct=5)
        self.assertEqual((free, forced), (2, 3))

    def test_degraded_all_free(self):
        free, forced = self._alloc(8, 3, 2, distinct=0)
        self.assertEqual((free, forced), (8, 0))


if __name__ == "__main__":
    unittest.main()
