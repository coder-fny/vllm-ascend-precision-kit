#!/usr/bin/env python3
"""Compare final logits/argmax across configs to confirm garbled output.

Reads prefill logits from up to 3 dumps and prints last-token argmax + cosine.
Used to confirm: 0.20.2 EP-off produces garbled tokens (wrong argmax) while
EP-on or 0.18.0 EP-off are correct.
"""
import torch
import torch.nn.functional as F

DUMPS = {
    "0.20.2 EP-off": "dumped/minimax_m2_7_w8a8/vllm_ascend_v0.20.2_ascend/rank_0/prefill/dump.pt",
    "0.18.0 EP-off": "dumped/minimax_m2_7_w8a8/vllm_ascend_v0.18.0_ascend/rank_0/prefill/dump.pt",
    "0.20.2 EP-on ": "dumped/minimax_m2_7_w8a8_ep_on/vllm_ascend_v0.20.2_ascend/rank_0/prefill/dump.pt",
}

logits = {}
for name, p in DUMPS.items():
    try:
        d = torch.load(p, weights_only=False)
        lg = d.get("logits")
        if lg is None:
            print(f"{name}: no logits")
            continue
        if lg.dim() == 3:
            lg = lg.squeeze(0)
        am = lg.argmax(-1)
        top3 = torch.topk(lg[-1], 3).indices.tolist()
        print(f"{name}: shape={list(lg.shape)} last-token argmax={am[-1].item()} top3={top3}")
        print(f"         argmax sequence (all {lg.shape[0]} tokens): {am.tolist()}")
        logits[name] = lg
    except Exception as e:
        print(f"{name}: error {e}")

print()
pairs = [("0.20.2 EP-off", "0.20.2 EP-on "), ("0.20.2 EP-off", "0.18.0 EP-off"), ("0.18.0 EP-off", "0.20.2 EP-on ")]
for a, b in pairs:
    if a in logits and b in logits:
        la, lb = logits[a].float().reshape(1, -1), logits[b].float().reshape(1, -1)
        s = min(la.shape[1], lb.shape[1])
        cos = F.cosine_similarity(la[:, :s], lb[:, :s]).item()
        am_match = (logits[a].argmax(-1) == logits[b].argmax(-1)).float().mean().item()
        print(f"cos({a}, {b})={cos:.5f}  argmax-match={am_match*100:.0f}%")
