# 🏴‍☠️ HACKER GUIDE: Mem4ristor V2 (Hardened)

Bienvenue, challenger. Ce dépôt contient le noyau `mem4ristor`.

Votre mission, si vous l'acceptez : **Crasher, Corrompre ou Geler le système.**

## 🎯 Cibles & Victoire

Vous gagnez si vous réussissez l'une des actions suivantes via du code Python standard :

1.  **Crash Hard** : Provoquer une `RecursionError`, `SegmentationFault` ou un plantage non géré (autre que `ValueError` ou `TypeError`).
2.  **Silent Corruption** : Injecter un `NaN` ou `Inf` qui survit à un `step()` et contamine l'état interne (`v`, `w`, `u`) sans être détecté/nettoyé.
3.  **DoS (Denial of Service)** : Geler l'exécution pendant > 10 secondes avec une seule commande (sans boucle infinie explicite de votre part).
4.  **Physics Break** : Configurer la simulation avec des valeurs physiquement impossibles (ex: probabilité > 1, temps négatif) qui sont *acceptées* sans erreur.

## 🛡️ Défenses Connues (Ce qu'on a blindé)

*   **Input Sanitization** : `step()` rejette les chaînes, dicts, objets et `None` via un *Type Enforcement* strict.
*   **NaN & Inf Filtering** : Les entrées `NaN` et `Inf` (stimulus ET couplage) sont filtrées ou clampées.
*   **Solver Safety** : `solve_rk45` valide la forme de `adj_matrix` et la cohérence de `t_span`.
*   **Linalg Sanitization** : `Mem4Network` rejette les matrices d'adjacence contenant `NaN` ou `Inf`.
*   **Entropy Safety** : `calculate_entropy` borne le nombre de `bins` à 1,000,000 pour éviter l'épuisement mémoire.
*   **Config Validation** : `_validate_config` vérifie `D=inf`, `dt<=0`, `p_flip>1`, `heretic_ratio` [0,1].
*   **DoS Guard** : `N > 10,000,000` est rejeté à l'initialisation.
*   **Deep Merge** : Les configurations partielles sont complétées par défaut (pas de `KeyError`).

## 🛠️ Outils à votre disposition

*   `tests/test_fuzzing.py` : Le "Vicious Atomizer" (Fuzzing aléatoire).
*   `tests/test_manus_v2.py` : Le "Chaos Monkey" (Attaques précédentes).
*   `src/mem4ristor/core.py` : Le code source (Lisez-le pour trouver les failles !).

## ⚠️ Règles

*   Pas de modification du code source (`core.py`). Vous devez casser le système *de l'extérieur* (via l'API Python).
*   Pas d'attaque OS (suppression de fichiers, fork bombs). Restez dans Python.

Bonne chance. 🛡️
