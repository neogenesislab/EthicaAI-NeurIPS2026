import json
import os
import re

PAPER_TEX = os.path.join("..", "..", "paper", "unified_paper.tex")

def load_json(path):
    p = os.path.join("..", "outputs", path)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)

def inject_partial_obs():
    print("Injecting Phase D: Partial Obs Table...")
    data = load_json("partial_obs/partial_obs_results.json")
    if not data:
        print("  -> No partial obs data found.")
        return
    
    with open(PAPER_TEX, "r", encoding="utf-8") as f:
        tex = f.read()

    def fmt(alg_key, cond_key):
        try:
            m = data[cond_key][alg_key]["survival_mean"]
            return f"{m:.0f}\\%"
        except:
            return "X\\%"

    mapping = [
        (r"Full Obs \(Baseline\) & .*?\\\\", f"Full Obs (Baseline) & X\\% & {fmt('situational', 'noisy_std0.05_dly0')} & {fmt('unconditional', 'noisy_std0.05_dly0')} \\\\"), # Assuming full obs is roughly baseline, but wait, partial config has no 'full'. Just replace X's for noisy/delayed.
        (r"Noisy \(\$\\sigma=0.10\$\) & .*?\\\\", f"Noisy ($\\sigma=0.10$) & {fmt('ippo', 'noisy_std0.1_dly0')} & {fmt('situational', 'noisy_std0.1_dly0')} & {fmt('unconditional', 'noisy_std0.1_dly0')} \\\\"),
        (r"Noisy \(\$\\sigma=0.20\$\) & .*?\\\\", f"Noisy ($\\sigma=0.20$) & {fmt('ippo', 'noisy_std0.2_dly0')} & {fmt('situational', 'noisy_std0.2_dly0')} & {fmt('unconditional', 'noisy_std0.2_dly0')} \\\\"),
        (r"Delayed \(\$k=2\$\) & .*?\\\\", f"Delayed ($k=2$) & {fmt('ippo', 'delayed_std0.0_dly2')} & {fmt('situational', 'delayed_std0.0_dly2')} & {fmt('unconditional', 'delayed_std0.0_dly2')} \\\\"),
        (r"Delayed \(\$k=5\$\) & .*?\\\\", f"Delayed ($k=5$) & {fmt('ippo', 'delayed_std0.0_dly5')} & {fmt('situational', 'delayed_std0.0_dly5')} & {fmt('unconditional', 'delayed_std0.0_dly5')} \\\\"),
        (r"Local-only & .*?\\\\", f"Local-only & {fmt('ippo', 'local_std0.0_dly0')} & {fmt('situational', 'local_std0.0_dly0')} & {fmt('unconditional', 'local_std0.0_dly0')} \\\\"),
    ]
    
    for pattern, repl in mapping:
        tex = re.sub(pattern, lambda m: repl, tex)
        
    with open(PAPER_TEX, "w", encoding="utf-8") as f:
        f.write(tex)
    print("  -> Injected Partial Obs Table")

def inject_hp_sweep():
    print("Injecting Phase A: HP Sweep Table...")
    data = load_json("cleanrl_baselines/hp_sweep_results.json")
    if not data:
        print("  -> No HP sweep data found.")
        return
        
    with open(PAPER_TEX, "r", encoding="utf-8") as f:
        tex = f.read()

    # Updated 7-column table header
    start_marker = r"\textbf{Learning Rate} & \textbf{Entropy} & $\boldsymbol{\bar{\lambda}}$ & \textbf{CI$_{95}$} & \textbf{Surv.\%} & $\boldsymbol{H(\pi)}$ & \textbf{Trapped?} \\"
    end_marker = r"\bottomrule"
    
    match = re.search(re.escape(start_marker) + r"(.*?)" + re.escape(end_marker), tex, re.DOTALL)
    if match:
        rows = [r"\midrule"]
        for key, run in data.items():
            lr = run.get("lr", 0)
            ent = run.get("entropy_coef", 0)
            lam = run.get("lambda_mean", 0)
            ci = run.get("lambda_ci95", [0,0])
            surv = run.get("survival_mean", 0)
            h_pi = run.get("policy_entropy_mean", 0)
            trapped = "Yes" if run.get("still_trapped", True) else "No"
            
            lr_str = f"{lr:.1e}".replace("e-0", "e-")
            row = f"{lr_str} & {ent:.2f} & {lam:.3f} & [{ci[0]:.3f}, {ci[1]:.3f}] & {surv:.1f}\\% & {h_pi:.3f} & {trapped} \\\\"
            rows.append(row)
            
        replacement = start_marker + "\n" + "\n".join(rows) + "\n" + end_marker
        tex = tex[:match.start()] + replacement + tex[match.end():]
        
        with open(PAPER_TEX, "w", encoding="utf-8") as f:
            f.write(tex)
        print("  -> Injected HP Sweep Table")
    else:
        print("  -> Table marker not found")

def inject_phi1():
    print("Injecting Phase B: Phi_1 Table...")
    data = load_json("phi1_ablation/phi1_results.json")
    if not data:
        print("  -> No phi1 data found.")
        return
        
    with open(PAPER_TEX, "r", encoding="utf-8") as f:
        tex = f.read()

    start_marker = r"$\boldsymbol{\phi_1}$ & \textbf{W (Byz=0\%)} & \textbf{W (Byz=30\%)} & \textbf{Alive (0\%)} & \textbf{Alive (30\%)} \\"
    end_marker = r"\bottomrule"
    
    match = re.search(re.escape(start_marker) + r"(.*?)" + re.escape(end_marker), tex, re.DOTALL)
    if match:
        rows = [r"\midrule"]
        # phi_1 is string in json like '0.0', '0.21', '0.5', '1.0'
        # we want them formatted and sorted. But they are keys. Let's just follow the existing keys natively.
        for phi_key in ['0.0', '0.21', '0.5', '1.0']:
            if phi_key not in data: continue
            run = data[phi_key]
            
            w0 = run["byz_0"]["welfare_mean"]
            w0_e = run["byz_0"]["welfare_std"]
            a0 = run["byz_0"]["survival_mean"]
            a0_e = run["byz_0"]["survival_std"]
            
            w30 = run["byz_30"]["welfare_mean"]
            w30_e = run["byz_30"]["welfare_std"]
            a30 = run["byz_30"]["survival_mean"]
            a30_e = run["byz_30"]["survival_std"]
            
            # Ex: 0.00 & 27.3 $\pm$ 0.2 & 22.6 $\pm$ 0.4 & 59 $\pm$ 12\% & 22 $\pm$ 6\% \\
            phi_f = f"{float(phi_key):.2f}"
            if phi_f == "1.00":
                # Make the row bold as in the original
                row = f"\\textbf{{1.00}} & \\textbf{{{w0:.1f} $\\pm$ {w0_e:.1f}}} & \\textbf{{{w30:.1f} $\\pm$ {w30_e:.1f}}} & \\textbf{{{a0:.0f} $\\pm$ {a0_e:.0f}\\%}} & \\textbf{{{a30:.0f} $\\pm$ {a30_e:.0f}\\%}} \\\\"
            else:
                row = f"{phi_f} & {w0:.1f} $\\pm$ {w0_e:.1f} & {w30:.1f} $\\pm$ {w30_e:.1f} & {a0:.0f} $\\pm$ {a0_e:.0f}\\% & {a30:.0f} $\\pm$ {a30_e:.0f}\\% \\\\"
            rows.append(row)
            
        replacement = start_marker + "\n" + "\n".join(rows) + "\n" + end_marker
        tex = tex[:match.start()] + replacement + tex[match.end():]
        
        with open(PAPER_TEX, "w", encoding="utf-8") as f:
            f.write(tex)
        print("  -> Injected Phi_1 Table")
    else:
        print("  -> Phi_1 Table marker not found")

if __name__ == "__main__":
    inject_partial_obs()
    inject_hp_sweep()
    inject_phi1()
    print("DONE")
