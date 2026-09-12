# Desktop bluescreen during the readout sweep, 11 September 2026

## What happened

The overnight `run_readout_sweep.sh` finished seven SimpleQA models, then
started mistral-small-24b at 8-bit (24.9 GB on the 32 GB RTX PRO 4500; every
earlier run was 16.3 GB or less). Ninety seconds into inference the NVIDIA
driver (nvlddmkm 595.79) logged errors and WHEA recorded corrected PCIe errors
on the GPU; at 05:38 the machine bluescreened with bugcheck 0x116
VIDEO_TDR_FAILURE (argument 3 STATUS_INSUFFICIENT_RESOURCES). Windows Update
then rebooted twice more for KB5124008, and the box was unreachable until
13:40. The same PCIe corrected errors appeared at the 05:40 boot before any
load, so the link itself (Gen 5 x16) is suspect, not only the workload.

Lost with the reboot: the mistral-small-24b run, and the chained background
jobs waiting on it (qwen2.5-32b 4-bit, the ARC-Challenge sweep, the steering
chain, the DeepSeek no-think re-run). The deepseek-r1-14b result in
`data/results/readout/invalid/` is unusable: every option pass hit the
generation fallback because R1 opens with a think block.

## Mitigations now in place

| Change | Where |
|---|---|
| mistral-small-24b runs at 4-bit (about 13 GB) | `PRECISION_OVERRIDES` in `run_readout_sweep.sh` |
| Unbuffered Python logs, stale partials removed before a retry | `run_readout_sweep.sh` |
| Crash-loop guard: a job is started at most twice | `data/logs/readout_attempts.txt` |
| One resumable queue for all remaining work, disarms when done | `run_readout_queue.sh` |
| Queue resumes 2 min after any logon (Remote Desktop included) | scheduled task `NNConfReadoutQueue`, `jobs/register_readout_queue_task.ps1` |
| GPU watchdog 60 s instead of 2 s, RDP and no-reboot policy pinned | `jobs/harden_gpu_host.ps1` (admin, reboot once) |

The 150 W power cap that first went with those settings was reverted on 12 September
2026 (`jobs/restore_gpu_power.ps1`). It was precautionary and the evidence did not
support it: the 7.5 h uncapped queue drew up to 186 W with zero nvlddmkm and zero
WHEA events, the corrected PCIe errors also appear at an idle boot with no GPU load,
and the bugcheck argument was STATUS_INSUFFICIENT_RESOURCES. The card draws 186 W on
a 24B 4-bit run, so the cap was a real 19 % clamp for no measured benefit. The VRAM
ceiling and the watchdog delay are the mitigations that address the actual failure.

## Outcome of the re-run, 11-12 September 2026

The queue ran unattended from 16:16 to 23:47 on 11 September and completed every
step with no driver error, no WHEA event and no reboot: mistral-small-24b (4-bit),
qwen2.5-32b (4-bit), deepseek-r1-14b with the empty think block, ARC-Challenge on
four models, and steering on three. Peak VRAM was 19.2 GB (qwen2.5-32b), against
the 24.9 GB that crashed the machine.

Two load figures worth keeping in mind: mistral-nemo-12b at FP16 reaches 24.5 GB,
the same region as the crash, and 8-bit for anything at or above 24B parameters
does too. Prefer 4-bit above about 14B on this card.

## Storage layout, 12 September 2026

| Drive | Device | Free before | Holds |
|---|---|---|---|
| D | Crucial MX500 SATA SSD, 931 GB | 67.3 GB | the repo, run output, 207 GB of weights |
| E | Transcend 4 TB external USB SSD | 2264.8 GB | 146 GB of weights incl. Qwen2.5-32B |

`jobs/consolidate_model_cache.ps1` moves the HuggingFace cache to E and leaves a
directory junction at `D:\git\NNConfidence\data\models`, so the hardcoded
`D:/git/NNConfidence/data/models` defaults in `model_loader.py` and
`extract_layer_signals.py` keep resolving. It copies, verifies every file by size,
and only then deletes the D copy, so a USB dropout mid-copy costs nothing.

Run output (`data/results`, 8.4 GB, about 1 GB of residuals per model) deliberately
stays on the internal drive: it is written continuously through multi-hour jobs,
where a USB dropout would kill the run. Weights are read-mostly, so E suits them.
With the junction in place, a model load needs E present at that drive letter.

## Still to consider

- BIOS: force the GPU slot to PCIe Gen 4. Corrected PCIe errors at idle boot
  are the signature of a marginal Gen 5 link.
- Reseat the card and its power connector.
- Clean reinstall of the NVIDIA driver if 0x116 recurs.
- The minidump is `C:\WINDOWS\Minidump\091126-14234-01.dmp` (admin to read).
