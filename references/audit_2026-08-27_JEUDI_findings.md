# JEUDI 2026-08-27 — Scripts d'experience findings

**Commit audite:** 9995db6 (V6.0.0, EN SYNC avec origin/main, 0 commit de retard — post-fetch)
**Branche:** main
**Interprete Python:** C:/Users/julch/AppData/Local/Programs/Python/Python313/python.exe (numpy 2.2.6 OK)
**Pool scripts:** 101 (experiments/*.py) — base reelle a 2026-08-27
**Baseline figures/:** 140 fichiers (90 csv + 49 png + autres) — evolution vs baseline skill historique (83) = +57 fichiers depuis la vague V6.0.0 (b-series, expA/expB, b6 sweep SPICE)

## Resultats

### Echantillon randomise (7 scripts, shuf)
Compile OK :
1. `spice_art_kirchhoff.py` — OK
2. `doubt_compute_allocation_poc.py` — OK
3. `verify_pb_isolated_node.py` — OK
4. `ngspice_locator.py` — OK
5. `event_phase_transition.py` — OK
6. `expB7_ceiling_control_poc.py` — OK
7. `p15_lorenz_chaos_poc.py` — OK

### Sweep complet
101/101 scripts compilent sans erreur syntaxique. **0 erreur, 0 warning notable.**

### Cross-check README <-> scripts (recette valide)
Sur le sous-ensemble verifie, les scripts produisent les artifacts aux bons noms, MAIS certains PNG manquent dans figures/ (ex: `p2_art_benchmark.png`, `p2_doubt_community_detection.png`, `p2_tau_u_bifurcation.png`, `p2_edge_betweenness.png`). Les CSV correspondants sont presents. Pas de bug code — pas reexecute depuis le dernier commit. Gravity = MINEUR.

### Pitfall CWD-relative (herite de juin 2026)
SEUL legacy reliquat detecte : `experiments/reviewer2_linear_stability.py` (mai 2026, archive vivant). Utilise `df.to_csv('reviewer2_linear_stability.csv', ...)` et `plt.savefig('reviewer2_linear_stability.png', ...)` SANS prefixe `ROOT / "figures"`. Si execute depuis le CWD racine, pollue la racine.
- **Pollution reelle aujourd'hui : 0** (root du repo inspecte — aucun `reviewer2_*` ou `protocole_*` au top level)
- Mais le piege dormant existe et resurgira au prochain `run` de ce script sans `cd experiments/`.
- Les 2 scripts `run_heroic_1600.py` / `run_heroic_800.py` utilisent `'../figures/v6_binder_cumulant_U4.csv'` et `'../figures/v6_binder_cumulant.png'` (relatif parent) — egalement piege si execute depuis racine, mais `figures/v6_binder_cumulant*` EXISTE deja au bon endroit.
- Tous les scripts recents (b1-b6, expA/expB, p15-p20, lambda2_foundation/*) utilisent systematiquement `ROOT / "figures"` ou `FIG_DIR = ROOT / "figures"`.
- Gravity = MINEUR (legacy inerte tant que pas reexecute)

### DECOUVERTE MAJEURE — Couverture README catastrophique
**77 / 101 (76%) des scripts experiments/ NE SONT PAS mentionnes dans le README.**
- 24 scripts mentionnes (la liste `p2_*` historique + `spice_*` + `v6_binder_cumulant_u4` + `run_heroic_*` + Binder SPICE)
- 77 scripts orphelins (toute la vague V6.0.0 : b1-b6 series complet, expA/expB, p15-p20, lambda2_foundation, doubt_compute_allocation_poc, event_phase_transition, etc.)
- Le README reste fige sur le périmètre legacy p2_, alors que le repo a evolue vers V6.0.0 avec 4 vagues d'experiences non-documentees.
- **Gravity = MAJEUR** (visibilite externe du repo degradee pour tout nouveau lecteur / reviewer / Julien lui-meme dans 3 mois).

### Pitfall SyntaxWarning
Aucun script ne declenche `SyntaxWarning: invalid escape sequence` (les f-strings LaTeX ont ete nettoyes, OU le grep `\l` n'a rien attrape). Aucun a signaler.

## Inventaire legacy dans repo
- `references/` untracked : contient `audit_2026-08-25_MARDI_findings.md` + `audit_2026-08-26_MERCREDI_findings.md` — pattern d'archivage des audits, OK
- `.commit-msg-pending.md` tracked mais ancien (3 juin) — probablement un brouillon oublie. Pas pollueur.
- Aucun `.csv`/`.png` au root, aucun dossier experiments/footnote non-standard.

## Synthese
- Compilation : OK ✅
- SyntaxWarning : OK ✅
- CWD pollution : 0 actif, 3 scripts dormants ⚠️
- Cross-check artifacts : quelques PNG absents (non-regression, pas regeneres) ⚠️
- Couverture README : 24/101 (24%) — DECOUVERTE MAJEURE ⚠️⚠️

## Signatures canoniques (rappel, transfere de MARDI 25/08)
- `Mem4ristorV3(config=None, seed=42)` ✓
- `Mem4Network(size=10, heretic_ratio=0.15, seed=42)` ✓
- `SensoryFrontend(output_dim, ...)` ✓
- `CreativeProjector(mem4ristor_instance, num_classes, ...)` ✓
- `DreamVisualizer(sensory_frontend, ...)` ✓
