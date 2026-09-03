# LUNDI 2026-08-31 — README + Badges findings

**Commit audite:** 9995db6 (V6.0.0, working tree PROPRE, en sync avec origin/main — `0 0` apres `git fetch`)
**Branche:** main
**Repo local:** `/d/ANTIGRAVITY/GITHUB_REPOSITORY/Mem4ristor-main` (skill historique pointait `mem4ristor-v2-main` — obsolete depuis le reorg V6.0.0)

## Resultats

### Version canonique
- `src/mem4ristor/__init__.py:58` : `__version__ = "6.0.0"`
- `pyproject.toml:7` : `version = "6.0.0"`
- README ligne 1 + status line : V6.0.0
- **3 sources coherentes — OK**

### DOI Zenodo — alignement post-audit du 02/08
Le repo a **3 DOI distincts** qui sont maintenant **tous legitimes** (et non plus contradictoires) :

| DOI | Emplacement | Role |
|-----|-------------|------|
| `10.5281/zenodo.18620596` | README badge l.5, CITATION.cff, README bibtex, COMPENDIUM | **Concept DOI** — resout vers la derniere version deposee |
| `10.5281/zenodo.19986042` | README l.14, CITATION.cff l.10, PROJECT_STATUS.md l.83, CONTEXT.md | **DOI de version V4.0.0** (derniere release Zenodo, 2026-05-02) |
| `10.5281/zenodo.19700749` | UNIQUEMENT dans docs/TEST_HERMES.md (archive) + tests/test_version_consistency.py (test anti-regression) | **Reference historique v3.2.0 (avril 2026)** — intentionnellement preservee pour guard |

**Verdict** : L'audit du 02/08/2026 (`AUDIT_EXTERNE_2026-08-02.md`) a identifie la divergence et l'a resolue. README et CITATION.cff utilisent maintenant le **concept DOI canonique** (18620596) et explicitent le DOI de version V4.0.0 dans le status. Le test `test_version_consistency.py` veille a ce que le badge ne retombe pas sur le 19700749.

**Cross-check obligatoire (gravite historique)** :
- README/CITATION pointent vers `cafe-virtuel/Mem4ristor` (correct) — **0 occurrence `Jusyl236`** dans README, CITATION.cff, ou PROJECT_STATUS.md
- Les 3 occurrences `Jusyl236` restantes sont dans `archives/dashboard_premium.py`, `CONTRIBUTING.md`, et les backups historique du preprint (`docs/papers/preprint/history/`) — references historiques legitimes, pas le repo principal

**Gravity = RESOLU** (ancien BLOQUANT, desactive depuis le 02/08/2026).

### Chemins de demos
- `experiments/demo_chimera.py` : ✅ existe, reference README l.89
- `examples/demo_applied.py` : ✅ existe, reference README l.93
- Pas de dossier `experimental/` (le skill historique le mentionnait) — **OBSOLETE**, confirme disparition

### Key Scientific Features (README l.23-37) vs code
10 features listees dans README. Verification rapide : `ART`, `Metacognitive Plasticity`, `Compartimentalised`, `Non-Local Coupling`, `Sparse CSR`, `Levitating Sigmoid`, `Degree-Normalized Coupling` — toutes implementées dans `src/mem4ristor/` (reorg V6.0.0).
- Sub-pitfall `coupling_norm` : README l.102 liste `uniform`, `degree_linear`, `spectral` — le code implemente **6 modes** (`uniform`, `degree`, `degree_linear`, `degree_log`, `degree_power`, `spectral`). **Gravity = MINEUR** (README incomplet, pas faux).
- Sub-pitfall Levitating Sigmoid : README dit `tanh(π(0.5-u)) + δ` (canonique) — confirme coherent avec doc.

### `sonification.py` dans README
- Test : `for f in src/mem4ristor/*.py; do grep -q "$(basename $f)" README.md || echo "NOT IN README: $f"; done`
- Seul `__init__.py` est "NOT IN README" (normal, c'est l'aggregateur).
- `sonification.py` EST reference (sub-pitfall historique RESOLU depuis le reorg V6.0.0).

### Figure count baseline
- `ls figures/ | wc -l` : **140** (vs baseline historique 83 du 11 juin)
- Le saut reflete les nombreux scripts d'audit/critique executes durant l'ete 2026 (V6.0.0 alignment, ART, SPICE Kirchhoff, etc.)
- Pas de chute anormale detectee

## Conclusion

**OK — README + Badges : alignes post-audit du 02/08.** Working tree en sync avec origin/main. DOI Zenodo coherent (concept + version explicites), 0 URL `Jusyl236` dans les fichiers de presentation, demos aux bons chemins, features listees coherentes avec le code, 140 figures dans `figures/`.

**Points mineurs (non-bloquants) :**
- README § Configuration ne liste que 3 modes de `coupling_norm` sur 6 implementes (cf. `topology.py`)
- `sonification.py` toujours absent du `__all__` (cf. MARDI 25/08 — Julien l'a ajoute au README mais pas au `__all__`, omis cosmetique)

**Aucun BLOQUANT detecte ce LUNDI.**

## Comparaison baselines
- vs LUNDI 2026-06-08 (3 DOI divergents, URL Jusyl236 dans CITATION + PROJECT_STATUS) : **5 BLOQUANTS resolus** sur 5
- vs LUNDI spec skill historique : obsolète sur `chemin repo` et `version` (V3.2.0 → V6.0.0) — adapte au contexte reel

## Signatures canoniques confirmees (cf. MARDI 25/08)
- `Mem4ristorV3(config=None, seed=42)` ✓
- `Mem4Network(size=10, heretic_ratio=0.15, seed=42)` ✓
