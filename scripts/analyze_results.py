import json, os
os.chdir("/mnt/d/00.test/PAPER/EthicaAI_anon2")

# 1. Research Agenda Summary
a = json.load(open("experiments/evolution/research_agenda.json", encoding="utf-8"))
print("=" * 60)
print("RESEARCH AGENDA SUMMARY")
print("=" * 60)
print(f"Total Generations Run: {a['total_generations_run']}")
print(f"Questions Completed: {a['total_questions_completed']}")
print()
for k, v in a["questions"].items():
    print(f"  {k}: {v['status']:10s} | {v['type']:12s} | {v['question'][:60]}")

# 2. History Analysis
print()
print("=" * 60)
print("EXPERIMENT HISTORY ANALYSIS")
print("=" * 60)
h = json.load(open("experiments/evolution/history.json", encoding="utf-8"))
print(f"Total history entries: {len(h)}")

coops = []
for e in h:
    r = e.get("result", {})
    if r and "Prosocial" in r:
        c = r["Prosocial"].get("cooperation_rate", 0)
        coops.append(c)

if coops:
    print(f"Min cooperation: {min(coops):.4f}")
    print(f"Max cooperation: {max(coops):.4f}")
    print(f"Avg cooperation: {sum(coops)/len(coops):.4f}")
    print(f"Std cooperation: {(sum((c - sum(coops)/len(coops))**2 for c in coops)/len(coops))**0.5:.4f}")
    
    # Check if any exceeded 0.15
    above_15 = [c for c in coops if c > 0.15]
    print(f"Times > 0.15: {len(above_15)} / {len(coops)}")
    above_20 = [c for c in coops if c > 0.20]
    print(f"Times > 0.20: {len(above_20)} / {len(coops)}")

print()
print("Last 10 experiments:")
for e in h[-10:]:
    cfg = e.get("config", {})
    r = e.get("result", {})
    coop = r.get("Prosocial", {}).get("cooperation_rate", "N/A") if r else "N/A"
    mode = cfg.get("GENESIS_LOGIC_MODE", "?")
    ia = cfg.get("USE_INEQUITY_AVERSION", False)
    beta = cfg.get("GENESIS_BETA", "?")
    alpha = cfg.get("GENESIS_ALPHA", "?")
    platform = e.get("platform", "?")
    print(f"  coop={coop}, mode={mode}, beta={beta}, alpha={alpha}, IA={ia}, platform={platform}")

# 3. Mode distribution
print()
print("Mode Distribution:")
modes = {}
for e in h:
    m = e.get("config", {}).get("GENESIS_LOGIC_MODE", "unknown")
    modes[m] = modes.get(m, 0) + 1
for m, c in sorted(modes.items(), key=lambda x: -x[1]):
    print(f"  {m}: {c}")

# 4. IA usage
ia_on = sum(1 for e in h if e.get("config", {}).get("USE_INEQUITY_AVERSION", False))
print(f"\nIA Enabled: {ia_on} / {len(h)}")

# 5. Platform
platforms = {}
for e in h:
    p = e.get("platform", "unknown")
    platforms[p] = platforms.get(p, 0) + 1
print(f"Platforms: {platforms}")
