"""Durable, single-writer Spec F orchestration; one CUDA subprocess at a time."""
import fcntl
import hashlib
import json
import os
import statistics
from pathlib import Path
import shutil
import subprocess
import sys
import time

OUT = Path('/workspace/spec_f_outputs')
ARMS = ['baseline', 'SFT-twin', 'DPO', 'grad-diff']


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def event(**value):
    value = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **value}
    with (OUT / 'ledger.jsonl').open('a') as f:
        f.write(json.dumps(value) + '\n')
        f.flush()
        os.fsync(f.fileno())
    with (OUT / 'RUN_LOG.md').open('a') as f:
        f.write('\n- ' + value['utc'] + ' ' + json.dumps(value) + '\n')
    print(json.dumps(value), flush=True)


def report():
    gates = {}
    for arm in ARMS:
        path = OUT / arm / 'gate/complete.json'
        if path.exists():
            gates[arm] = json.loads(path.read_text())
    values = {}
    baseline = gates.get('baseline')
    for arm in ARMS:
        gate = gates.get(arm)
        values[arm] = {'status': 'available' if gate else 'pending_or_parked',
                       'delta_mean_real_probability':
                           gate['primary']['mean_real_probability'] - baseline['primary']['mean_real_probability']
                           if gate and baseline else None,
                       'metrics': gate}
    write(OUT / 'pre_attack_gate.json', {'definition': 'Change in mean normalized-choice probability of real answer versus baseline; negative means less real-answer belief, not by itself twin installation.',
                                       'primary_n': 15, 'direct_twin_n': 7, 'arms': values})
    lines = ['# Spec F preliminary results', '', '## Pre-attack belief shifts', '',
             '| Arm | Δ real-answer probability | Primary acc_norm | Twin probability (n=7) |',
             '|---|---:|---:|---:|']
    for arm, val in values.items():
        g = val['metrics']
        cells = [f"{val['delta_mean_real_probability']:+.6f}", f"{g['primary']['acc_norm']:.4f}",
                 f"{g['twin_mapped']['mean_twin_probability']:.6f}"] if g and baseline else ['pending / parked'] * 3
        lines.append('| ' + ' | '.join([arm] + cells) + ' |')
    lines += ['', '**Only 15 primary questions: one answer is 6.67 percentage points.** Full cloze n=1076. '
              'Direct twin-choice mapping exists for seven questions. These are exploratory results from mixed-generation documents with known coverage defects.',
              '', '## Attack checkpoints', '', '| Arm | Step | Primary acc_norm | Full acc_norm | Primary real probability |',
              '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        for path in sorted((OUT / arm / 'attack').glob('metrics_*.json')):
            m = json.loads(path.read_text())
            lines.append(f"| {arm} | {m['step']} | {m['primary']['acc_norm']:.4f} | {m['full']['acc_norm']:.4f} | {m['primary']['mean_real_probability']:.6f} |")
    lines += ['', '## Run details', '',
              '- 163 usable aligned pairs: 140 training, 23 validation; 24 training papers and three held-out papers. One context overflow dropped; no alignment failures within the 27 papers.',
              '- Installation: 64 optimizer steps, LR 5e-5; attack: 1000 steps with the exact requested logarithmic evaluation schedule.',
              '- All arms use the same cached real-data order. Three held-out papers are excluded from attack data. Plain LM text; empty DPO textual prompt plus shared boundary token.',
              '- One H100 is visible. Sequential CUDA processes with a ten-second settle delay.',
              '- FP32 weights, trainable parameters and gradients; BF16 autocast, SDPA. torch.compile skipped to avoid startup cost.',
              '- Every logged adapter and per-question FP32 score/probability table is retained. Scores are sequence log likelihoods, not single-token logits.',
              '- Grad-diff caps each original sequence mean NLL at 20 nats/token, then token-weights it. Every cap activation is logged; nonfinite loss/gradients park the arm.']
    lines += ['', '## Measured throughput and instability', '',
              '| Arm | Phase | Logged steps | Median seconds/step | Tokens/s overall | Cap activations |',
              '|---|---|---:|---:|---:|---:|']
    for arm in ARMS:
        for phase_name in ['install', 'attack']:
            ledger = OUT / arm / phase_name / 'ledger.jsonl'
            if not ledger.exists():
                continue
            steps = []
            for line in ledger.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # writer may currently be appending the final line
                if row.get('event') == 'step':
                    steps.append(row)
            if steps:
                seconds = sum(x['seconds'] for x in steps)
                rate = sum(x['tokens'] for x in steps) / seconds
                caps = sum(x['clamped_sequences'] for x in steps)
                lines.append(f"| {arm} | {phase_name} | {len(steps)} | {statistics.median(x['seconds'] for x in steps):.2f} | {rate:.1f} | {caps} |")
    batchpath = OUT / 'batch_configuration.json'
    if batchpath.exists():
        lines += ['', 'Actual batch configuration:', '', '```json', batchpath.read_text().strip(), '```']
    for arm in ARMS:
        parked = OUT / arm / 'parked.json'
        if parked.exists():
            lines += ['', f"**{arm} parked:** `{json.loads(parked.read_text())}`"]
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def phase(arm, name, env, worker):
    folder = OUT / arm / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        event(event='skip_completed', arm=arm, phase=name)
        return True
    if (OUT / arm / 'parked.json').exists():
        event(event='skip_parked', arm=arm, phase=name)
        return False
    # Never overlap another existing CUDA process. An occupied assigned card is
    # recorded and retried automatically, rather than risking context OOM.
    while True:
        check = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                               text=True, capture_output=True)
        if check.returncode != 0:
            raise RuntimeError('Cannot verify exclusive GPU availability: ' + check.stderr)
        if not check.stdout.strip():
            break
        event(event='waiting_for_exclusive_gpu', arm=arm, phase=name)
        time.sleep(30)
    if (folder / 'started.json').exists():
        parked = {'phase': name, 'reason': 'Prior phase interrupted; no completed artifact. Parked without silently restarting weights.'}
        write(OUT / arm / 'parked.json', parked)
        event(event='arm_parked', arm=arm, **parked)
        return False
    with (folder / 'worker.log').open('a') as log:
        proc = subprocess.Popen([sys.executable, '-u', str(worker), '--arm', arm, '--phase', name],
                                stdout=log, stderr=subprocess.STDOUT, env=env)
        write(folder / 'started.json', {'pid': proc.pid, 'phase': name, 'arm': arm})
        write(OUT / 'status.json', {'status': 'running', 'arm': arm, 'phase': name,
                                   'worker_pid': proc.pid, 'supervisor_pid': os.getpid()})
        event(event='phase_launched', arm=arm, phase=name, pid=proc.pid)
        while proc.poll() is None:
            time.sleep(15)
            report()
        code = proc.returncode
    event(event='phase_exited', arm=arm, phase=name, exit_code=code)
    success = code == 0 and (folder / 'complete.json').exists()
    if not success:
        parked = {'phase': name, 'exit_code': code, 'log': str(folder / 'worker.log')}
        write(OUT / arm / 'parked.json', parked)
        event(event='arm_parked', arm=arm, **parked)
    time.sleep(10)
    report()
    return success


def main():
    lock = (OUT / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = json.loads((OUT / 'manifest.json').read_text())
    for path, expected in cfg['source_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'HF_HUB_OFFLINE': '1',
           'TOKENIZERS_PARALLELISM': 'false', 'PYTHONDONTWRITEBYTECODE': '1',
           'PYTORCH_ALLOC_CONF': 'expandable_segments:True'}
    worker = Path(__file__).with_name('spec_f_run.py')
    event(event='supervisor_started', pid=os.getpid(), gpu_count=1)
    if not phase('baseline', 'gate', env, worker):
        write(OUT / 'status.json', {'status': 'blocked_shared_preflight', 'reason': 'Baseline gate establishes shared batches/reference.'})
        return
    for arm in ARMS[1:]:
        phase(arm, 'install', env, worker)
    report()
    event(event='pre_attack_gate_collected', path=str(OUT / 'pre_attack_gate.json'))
    for arm in ARMS:
        phase(arm, 'attack', env, worker)
    report()
    parked = [arm for arm in ARMS if (OUT / arm / 'parked.json').exists()]
    write(OUT / 'status.json', {'status': 'finished_with_parked_arms' if parked else 'complete', 'parked': parked})
    event(event='supervisor_finished', parked=parked)


if __name__ == '__main__':
    main()
