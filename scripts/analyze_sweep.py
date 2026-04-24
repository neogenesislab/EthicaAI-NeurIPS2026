#!/usr/bin/env python3
"""Full Sweep 결과 분석 — 학습 곡선 트렌드 확인."""
import json, os

results_dir = 'experiments/full_sweep_results'

for f in sorted(os.listdir(results_dir)):
    if not f.endswith('.json') or f == 'full_sweep_summary.json':
        continue
    path = os.path.join(results_dir, f)
    with open(path) as fh:
        data = json.load(fh)
    
    exp = data.get('experiment', f)
    compile_t = data.get('compile_time_sec', 0)
    
    print(f'\n=== {exp} (compile: {compile_t:.1f}s) ===')
    
    for cond_name, cond in data.get('conditions', {}).items():
        n_ok = cond.get('n_successful', 0)
        n_fail = cond.get('n_failed', 0)
        
        if n_ok > 0:
            coop = cond.get('mean_cooperation', 0)
            rew = cond.get('mean_reward', 0)
            gini = cond.get('mean_gini', 0)
            std_c = cond.get('std_cooperation', 0)
            
            # 학습 곡선 트렌드: 처음 vs 마지막
            seeds = cond.get('seeds', [])
            curves = []
            for s in seeds:
                if 'error' in s:
                    continue
                lc = s.get('learning_curve', {})
                r = lc.get('reward_mean', [])
                c = lc.get('cooperation_rate', [])
                if len(r) >= 2:
                    curves.append({
                        'r_start': r[0], 'r_end': r[-1], 'r_delta': r[-1] - r[0],
                        'c_start': c[0] if c else 0, 'c_end': c[-1] if c else 0,
                    })
            
            if curves:
                avg_r_delta = sum(c['r_delta'] for c in curves) / len(curves)
                avg_r_start = sum(c['r_start'] for c in curves) / len(curves)
                avg_r_end = sum(c['r_end'] for c in curves) / len(curves)
                trend = 'UP' if avg_r_delta > 0.001 else ('DOWN' if avg_r_delta < -0.001 else 'FLAT')
            else:
                avg_r_delta = avg_r_start = avg_r_end = 0
                trend = '?'
            
            print(f'  {cond_name:15s} | Coop={coop:.4f}+/-{std_c:.4f} | '
                  f'Reward: {avg_r_start:.4f} -> {avg_r_end:.4f} ({trend}, delta={avg_r_delta:+.4f}) | '
                  f'{n_ok}ok/{n_fail}fail')
        else:
            print(f'  {cond_name:15s} | ALL FAILED ({n_fail})')
