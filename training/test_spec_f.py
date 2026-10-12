"""Offline mathematical/data invariants for Spec F; no model downloads."""
import json
import math
import unittest
import numpy as np
import torch
from spec_f_run import OUT, bounded_original, collate, dpo_loss, score_arrays, sequence_sums


class LossTests(unittest.TestCase):
    def test_dpo_identical_pair(self):
        x = torch.tensor([-5., -9.], requires_grad=True)
        loss = dpo_loss(x, x, x.detach(), x.detach())
        self.assertAlmostEqual(float(loss.detach()), math.log(2), places=6)
        loss.backward()
        torch.testing.assert_close(x.grad, torch.zeros_like(x))

    def test_dpo_direction(self):
        chosen = torch.tensor([-8.], requires_grad=True)
        rejected = torch.tensor([-8.], requires_grad=True)
        loss = dpo_loss(chosen, rejected, chosen.detach(), rejected.detach())
        loss.backward()
        self.assertLess(float(chosen.grad), 0.)
        self.assertGreater(float(rejected.grad), 0.)

    def test_causal_padding(self):
        ids, mask = collate([np.array([0, 1, 2]), np.array([0, 2])], 'cpu')
        logits = torch.zeros(2, 3, 3, requires_grad=True)
        sums, count = sequence_sums(logits, ids, mask)
        torch.testing.assert_close(count, torch.tensor([2, 1]))
        torch.testing.assert_close(sums, -count * math.log(3))
        (-sums.sum()).backward()
        torch.testing.assert_close(logits.grad[1, 1:], torch.zeros_like(logits.grad[1, 1:]))

    def test_grad_diff_ascent_and_cap(self):
        sums = torch.tensor([-10., -60.], requires_grad=True)
        bounded, flags = bounded_original(sums, torch.tensor([2, 2]), 20.)
        self.assertEqual(flags.tolist(), [False, True])
        (-bounded.sum()).backward()
        torch.testing.assert_close(sums.grad, torch.tensor([1., 0.]))

    def test_cloze_normalization_and_storage(self):
        rows = [{'question_id': 'toy', 'choices': ['a', 'long', 'bb', 'ccc'],
                 'gold': 1, 'matched': True, 'twin_choice_index': 0}]
        metrics, arrays = score_arrays([[-1, -2, -3, -4]], rows)
        self.assertEqual(metrics['primary']['acc_norm'], 1.)
        self.assertEqual(metrics['primary']['acc'], 0.)
        for name in ['raw_loglikelihood', 'normalized_loglikelihood', 'raw_probability', 'normalized_probability']:
            self.assertEqual(arrays[name].dtype, np.float32)
        np.testing.assert_allclose(arrays['normalized_probability'].sum(1), 1.)

    def test_canonical_scorer_and_adapter_merge(self):
        from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXForCausalLM
        from peft import LoraConfig, get_peft_model
        from lm_eval.models.huggingface import HFLM
        from lm_eval.api.instance import Instance
        torch.set_num_threads(2)
        torch.manual_seed(0)
        model = GPTNeoXForCausalLM(GPTNeoXConfig(
            vocab_size=50304, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=4, rotary_pct=1.,
            max_position_embeddings=128, attention_dropout=0., hidden_dropout=0.))
        model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4,
                    target_modules=['query_key_value'], task_type='CAUSAL_LM'))
        for name, param in model.named_parameters():
            if 'lora_B' in name:
                with torch.no_grad():
                    param.fill_(.01)
        tok = AutoTokenizer.from_pretrained('/workspace/models/deep-ignorance/unfiltered', local_files_only=True)
        model.eval()
        ids = torch.tensor([[0, 1, 2]])
        with torch.no_grad():
            before = model(ids).logits
            merged = model.merge_and_unload(safe_merge=True)
            after = merged(ids).logits
        torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)
        scorer = HFLM(pretrained=merged, tokenizer=tok, backend='causal',
                       batch_size=2, max_length=128, softmax_dtype=torch.float32)
        reqs = [Instance(request_type='loglikelihood', doc={},
                         arguments=('Question: A\nAnswer:', x), idx=i,
                         metadata=('synthetic', 0, 1)) for i, x in enumerate([' B', ' C'])]
        actual = scorer.loglikelihood(reqs, disable_tqdm=True)
        self.assertEqual(len(actual), 2)
        for req, (score, _) in zip(reqs, actual):
            context, continuation = scorer._encode_pair(*req.args)
            tokens = torch.tensor([context + continuation])
            with torch.no_grad():
                logits = merged(tokens).logits.float().log_softmax(-1)
            expected = sum(float(logits[0, len(context) - 1 + j, t]) for j, t in enumerate(continuation))
            self.assertAlmostEqual(score, expected, places=4)

    def test_split_and_order(self):
        split = json.loads((OUT / 'document_split.json').read_text())
        self.assertFalse(set(split['train_doc_ids']) & set(split['validation_doc_ids']))
        rows = [json.loads(s) for s in (OUT / 'pairs.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 163)
        self.assertEqual(sum(r['split'] == 'train' for r in rows), 140)
        order = np.load(OUT / 'cache/attack_order.npy')
        np.testing.assert_array_equal(np.sort(order), np.arange(90464))
        np.testing.assert_array_equal(order, np.random.default_rng(0).permutation(90464))
        heldout = {int(x.rsplit(':', 1)[1]) for x in split['validation_doc_ids']}
        attack = [json.loads(s) for s in (OUT / 'attack_chunks.jsonl').read_text().splitlines()]
        self.assertFalse(heldout & {r['row_index'] for r in attack})


if __name__ == '__main__':
    unittest.main()
