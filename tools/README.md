# `tools/` — vérifier les chiffres de ce dépôt vous-même

Ce dossier contient l'appareil qui vérifie que **chaque valeur numérique publiée dans le
preprint est bien celle que produisent les données du dépôt**. Il ne vous demande pas de
nous croire : il recalcule.

> **Depuis le 3 septembre 2026 seulement.** Ces outils vivaient jusque-là hors du dépôt, sur
> la machine de l'auteur. Le preprint et le compendium se réclamaient d'une vérification
> automatique que personne d'autre ne pouvait lancer — ni même lire. C'est corrigé ici.

---

## En une commande

```bash
pip install -r requirements.txt && pip install -e .
python tools/preprint_guardian.py
```

Sortie attendue : `Claims verifiees : 22 / OK : 22 / BLOQUEES : 0`.
Le code de retour est **non nul** si un seul chiffre publié ne correspond plus à ses données.

---

## Les deux garde-fous, et ce que chacun sait faire

### `preprint_guardian.py` — le chiffre publié vient-il bien du CSV ?

Pour chacun des **22 claims** décrits dans [`../docs/CLAIMS_REGISTER.md`](../docs/CLAIMS_REGISTER.md),
il ouvre le CSV de `figures/`, recalcule la valeur, et la compare à ce qui est écrit dans le
preprint, avec la tolérance déclarée dans `claims_mapping.json`.

| commande | ce qu'elle fait |
|---|---|
| `python tools/preprint_guardian.py` | vérifie les 22 claims, rapport détaillé |
| `... --fast` | idem, sortie courte |
| `... --claim C18` | un seul claim |
| `... --self-test` | **contrôle positif** — voir plus bas |
| `... --html --output r.html` | rapport HTML |
| `... --install-hook` | installe le hook `pre-commit` dans *votre* clone |

### `tex_guardian.py` — le texte publié dit-il encore ce que disent les données ?

Le premier compare une **valeur** à un **CSV**. Celui-ci compare le **texte du `.tex`** à ses
**sources** : il vérifie que chaque nombre ancré est celui du claim (N1), qu'aucune valeur
remplacée ne traîne encore dans le texte (N3), que les scripts et dossiers cités sont
réellement versionnés, et que chaque CSV de claim a un producteur dans le dépôt (N4).

```bash
python tools/tex_guardian.py
```

---

## Le contrôle positif — pourquoi un feu vert ne prouve rien

```bash
python tools/preprint_guardian.py --self-test
```

Un garde-fou qui dit « tout va bien » ne prouve rien **tant qu'on n'a pas montré qu'il sait
dire non**. Le self-test rejoue les claims dérivés `C11b`/`C11c` contre un **témoin figé**
(le commit `6833cde`, état antérieur au réalignement de δ) et **exige qu'ils y bloquent**.

Il exige aussi que `C11` y rende **la même valeur qu'aujourd'hui** — c'est la démonstration
rejouable d'un angle mort réel : jusqu'au 6 août 2026, une spec de claim ne savait désigner
qu'une **cellule** de tableau. Quand la phrase publiée portait sur un *ratio*, le contrôle
s'ancrait sur la cellule voisine et **restait vert pendant que la phrase devenait fausse**.

---

## Les fichiers

| fichier | rôle |
|---|---|
| `preprint_guardian.py` | vérificateur des claims |
| `tex_guardian.py` | vérificateur texte ↔ sources |
| `claims_mapping.json` | pour chaque claim : CSV, colonne, filtre de ligne, valeur attendue, tolérance |
| `tex_anchors.json` | pour chaque nombre publié : où il est dans le `.tex`, de quel claim il vient |

`claims_mapping.json` est le fichier à lire pour comprendre **ce qui est vérifié et comment** —
il porte aussi, en clair, les notes des corrections passées.

---

## Ce que cet appareil ne fait pas

Il mesure la **reproductibilité**, pas la **justesse scientifique**. Il garantit qu'un chiffre
publié vient bien de ses données ; il ne dit rien de la pertinence de la grandeur mesurée. Un
exemple ouvert et documenté : la calibration de la complexité LZ
(cf. `docs/audits/2026-08-02/`) — les comparaisons à durée constante tiennent, les seuils
absolus sont en cours de révision.

Deux limites connues, écrites parce qu'elles ont été payées :

- **N4 vérifie qu'un producteur existe et est versionné, pas qu'il peut s'exécuter.** Une
  dépendance externe absente (par exemple `ngspice`, requis par le claim `C11`) lui est
  invisible.
- **N2 (couverture) n'est concluant dans aucun des deux sens** : un signalement peut être faux
  (les entiers nus sont invisibles au détecteur), et une absence de signalement ne prouve rien.
  Il est affiché comme **informatif**, jamais bloquant.

---

## Environnement

`requirements.txt` ne pose que des versions **planchers**. Les chiffres publiés ont été produits
avec numpy 2.2.6 / scipy 1.16.3 / pandas 2.3.1. Les garde-fous ont été rejoués le 3 septembre
2026 avec numpy 2.5.2 / scipy 1.18.1 / pandas 3.0.5 : **22/22 dans les deux environnements**.
