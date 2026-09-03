#!/usr/bin/env python3
r"""
preprint_guardian.py — CI Scientifique pour le projet Mem4ristor
Repository : D:\ANTIGRAVITY\GITHUB_REPOSITORY\Mem4ristor-main

Role : Verifier automatiquement que les valeurs numeriques publiees dans
preprint.tex et CLAIMS_REGISTER.md correspondent aux resultats calcules
par les scripts source ou aux CSV deja generes.

Fonctionnement :
  1. Lit claims_mapping.json pourconnaitre le mapping claim -> (csv, colonne, valeur_attendue)
  2. Lis le CSV correspondant et extraie la valeur reelle
  3. Compare avec la tolerance configurable
  4. Bloque si delta > tolerance

Usage :
  python preprint_guardian.py                    # verification complete
  python preprint_guardian.py --fast             # CSV uniquement, pas de re-run scripts
  python preprint_guardian.py --claim C11        # verifier une claim specifique
  python preprint_guardian.py --self-test        # controle positif des claims derives
  python preprint_guardian.py --html             # rapport HTML
  python preprint_guardian.py --watch            # mode surveillance (JSON, cronjob)
  python preprint_guardian.py --install-hook     # installe le pre-commit hook

Avant tout commit touchant preprint.tex : lancer ce script.
Si status = BLOQUE → corriger avant de committer.
"""

import re
import csv
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Console Windows en cp1252 : force UTF-8 pour eviter UnicodeEncodeError sur les
# caracteres comme Delta. Sans ca, le pre-commit hook ET le cron nightly plantent
# des qu'un print contient un symbole non-cp1252. (fix 2026-05-29)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


# --- Configuration ---
# Repo de code vivant depuis le rangement du ~2026-07-14 : GITHUB_REPOSITORY/Mem4ristor-main
# (TEST_HERMES/mem4ristor-v2-main supprime ; publication poussee sur github.com/cafe-virtuel/Mem4ristor).
# Re-pointe le 2026-07-17.
#
# 2026-09-03 -- LE GUARDIAN VIT DESORMAIS DANS LE DEPOT (tools/), et cette constante
# n'est plus un chemin de machine. Motif : jusqu'a aujourd'hui il vivait dans
# D:\ANTIGRAVITY\.brain\, hors du depot publie -- donc QUI CLONE NE POUVAIT VERIFIER
# AUCUN CLAIM. Le compendium se reclamait d'un appareil que le depot ne livrait pas.
# La racine est deduite de l'emplacement du script : tools/ est un enfant direct de
# la racine du depot. Surchargeable par MEM4_ROOT dans l'environnement (CI, worktree).
MEM4_ROOT = Path(os.environ.get("MEM4_ROOT") or Path(__file__).resolve().parent.parent)
FIGURES_DIR = MEM4_ROOT / "figures"
MAPPING_FILE = Path(__file__).parent / "claims_mapping.json"


class ClaimResult:
    def __init__(self, claim_id, description, expected, actual, delta,
                 tolerance, unit, status, csv_name, note=""):
        self.id = claim_id
        self.description = description
        self.expected = expected
        self.actual = actual
        self.delta = delta
        self.tolerance = tolerance
        self.unit = unit
        self.status = status  # OK | BLOQUE | ERROR | A_VERIFIER | PARTIEL
        self.csv_name = csv_name
        self.note = note

    def __repr__(self):
        return (f"<ClaimResult {self.id} expected={self.expected} "
                f"actual={self.actual} delta={self.delta} "
                f"status={self.status}>")


def load_mapping(path: Path) -> dict:
    if not path.exists():
        print(f"[ERREUR] Mapping non trouve : {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_csv_row(csv_path: Path, row_filter: dict | str) -> dict | None:
    """Lit un CSV et retourne la premiere ligne qui correspond au filtre."""
    if not csv_path.exists():
        return None
    with open(csv_path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if isinstance(row_filter, str):
                # Filtre special "ratio_FROZEN_FULL"
                if row_filter == "ratio_FROZEN_FULL":
                    if (row.get('ablation', '').upper() in ('FROZEN', 'FROZEN_U')
                            and row.get('distance', '') == '1'):
                        frozen_val = float(row['mi_mean'])
                        # chercher FULL correspondant
                        f.seek(0)
                        next(reader)
                        for r in reader:
                            if (r.get('ablation', '') == 'FULL'
                                    and r.get('distance', '') == '1'):
                                full_val = float(r['mi_mean'])
                                row['ratio'] = str(frozen_val / full_val)
                                return row
                continue
            match = True
            for k, v in row_filter.items():
                row_val = row.get(k, '').strip()
                if row_val != v:
                    match = False
                    break
            if match:
                return row
    return None


def get_all_csv_rows(csv_path: Path) -> list[dict] | None:
    if not csv_path.exists():
        return None
    with open(csv_path, newline='', encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pick_row(rows: list[dict], row_filter: dict) -> dict | None:
    for row in rows:
        if all(row.get(k, '').strip() == v for k, v in row_filter.items()):
            return row
    return None


def _cell(row: dict, column: str) -> float | None:
    """Lit une cellule numerique, insensible a la casse du nom de colonne."""
    for col in row.keys():
        if col.lower() == column.lower():
            try:
                return float(row[col])
            except (ValueError, TypeError):
                return None
    return None


def compute_derived(csv_path: Path, derived: dict) -> tuple[float | None, str]:
    """Calcule une grandeur DERIVEE de plusieurs lignes/colonnes d'un CSV.

    POURQUOI CETTE FONCTION EXISTE (2026-08-06)
    -------------------------------------------
    Une spec de claim ordinaire ne sait designer qu'UNE CELLULE : (csv, ligne,
    colonne). Quand le registre publie une grandeur qui n'est pas une cellule --
    un ratio entre deux lignes, un ecart entre deux ratios -- l'ancrage se
    rabattait sur la cellule la PLUS PROCHE. C'est exactement ce qui s'est
    produit pour C11 : le registre publiait le ratio ART_soft/V4, le Guardian
    ancrait `delta_pct`, et les deux ne coincidaient que par accident. Au
    realignement de leak_delta (0.05 -> 0.01, 2026-08-05) la colonne est restee
    a 0.0 tandis que le ratio passait de 1.490 a 1.134 : LE GARDE-FOU SERAIT
    RESTE VERT PENDANT QUE LE CLAIM DEVENAIT FAUX.

    La dependance est donc declaree ICI, dans le mapping, et non par un
    `if claim_id == ...` dans verify_claim (pattern de C06/C08) : un comportement
    attache a l'ID est invisible quand on lit claims_mapping.json, et c'est cette
    invisibilite qui a laisse l'ecart vivre trois mois.

    Types supportes :
      row_ratio          colonne[numerator] / colonne[denominator]
      row_ratio_gap_pct  ecart relatif SIGNE, en %, entre le meme ratio calcule
                         sur colonne_a et sur colonne_b : (r_a - r_b) / r_b * 100

    Retourne (valeur, detail_lisible). valeur=None si le calcul est impossible.
    """
    rows = get_all_csv_rows(csv_path)
    if rows is None:
        return None, "CSV introuvable"

    dtype = derived.get("type", "")
    num_f = derived.get("numerator", {})
    den_f = derived.get("denominator", {})
    num_row, den_row = _pick_row(rows, num_f), _pick_row(rows, den_f)
    if num_row is None or den_row is None:
        return None, f"ligne introuvable (num={num_f}, den={den_f})"

    def ratio(col: str) -> tuple[float | None, str]:
        a, b = _cell(num_row, col), _cell(den_row, col)
        if a is None or b is None:
            return None, f"colonne '{col}' absente ou non numerique"
        if b == 0:
            return None, f"denominateur nul sur '{col}'"
        return a / b, f"{a}/{b}"

    if dtype == "row_ratio":
        r, det = ratio(derived.get("colonne", ""))
        return (None, det) if r is None else (r, f"{det} = {r:.6f}")

    if dtype == "row_ratio_gap_pct":
        col_a, col_b = derived.get("colonne_a", ""), derived.get("colonne_b", "")
        r_a, det_a = ratio(col_a)
        r_b, det_b = ratio(col_b)
        if r_a is None:
            return None, det_a
        if r_b is None:
            return None, det_b
        if r_b == 0:
            return None, "ratio de reference nul"
        gap = (r_a - r_b) / r_b * 100
        return gap, (f"ratio[{col_a}]={r_a:.6f} ({det_a}) vs "
                     f"ratio[{col_b}]={r_b:.6f} ({det_b}) -> {gap:+.4f} %")

    return None, f"type derive inconnu : '{dtype}'"


def verify_claim(claim_id: str, spec: dict) -> ClaimResult:
    """Verifie une claim contre le CSV specifie dans le mapping."""
    csv_name = spec.get("csv", "")
    row_filter = spec.get("row_filter", {})
    column = spec.get("colonne", "")
    expected = spec.get("expected")
    tolerance = spec.get("tolerance", 0.05)
    unit = spec.get("unit", "")
    description = spec.get("description", "")
    note = spec.get("note", "")

    # Cas special ratio MI (C08)
    if claim_id == "C08":
        csv_path = FIGURES_DIR / csv_name
        row = get_csv_row(csv_path, "ratio_FROZEN_FULL")
        if row is None:
            return ClaimResult(claim_id, description, expected, None, None,
                              tolerance, unit, "ERROR", csv_name, "Row non trouvee")
        actual = float(row.get('ratio', 0))
        delta = abs(actual - expected) if expected else None
        status = "OK" if (delta is not None and delta <= tolerance) else "BLOQUE"
        return ClaimResult(claim_id, description, expected, actual, delta,
                          tolerance, unit, status, csv_name, note)

    # Cas special alpha_crit (C06) — chercher la ligne ou re_lambda_max passe par 0
    if claim_id == "C06":
        csv_path = FIGURES_DIR / csv_name
        if not csv_path.exists():
            return ClaimResult(claim_id, description, expected, None, None,
                              tolerance, unit, "ERROR", csv_name)
        with open(csv_path, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # Trouver le alpha ou re_lambda_max est le plus proche de 0
        best = min(rows, key=lambda r: abs(float(r.get('re_lambda_max', 999))))
        actual = float(best['alpha'])
        delta = abs(actual - expected) if expected else None
        status = "OK" if (delta is not None and delta <= tolerance) else "BLOQUE"
        return ClaimResult(claim_id, description, expected, actual, delta,
                          tolerance, unit, status, csv_name, note)

    # Grandeur DERIVEE declaree dans le mapping (ratio entre lignes, ecart de
    # ratios...). Voir compute_derived() pour le pourquoi. Passe AVANT le cas
    # general : une spec derivee n'a ni row_filter ni colonne uniques.
    if spec.get("derived"):
        csv_path = FIGURES_DIR / csv_name
        actual, detail = compute_derived(csv_path, spec["derived"])
        if actual is None:
            return ClaimResult(claim_id, description, expected, None, None,
                               tolerance, unit, "ERROR", csv_name,
                               f"grandeur derivee non calculable : {detail}")
        delta = abs(actual - expected) if expected is not None else None
        if expected is None:
            status = "A_VERIFIER"
        else:
            status = "OK" if delta <= tolerance else "BLOQUE"
        return ClaimResult(claim_id, description, expected, actual, delta,
                           tolerance, unit, status, csv_name,
                           f"[derive] {detail} | {note}")

    # Cas general
    csv_path = FIGURES_DIR / csv_name
    row = get_csv_row(csv_path, row_filter) if row_filter else None

    if row is None:
        return ClaimResult(claim_id, description, expected, None, None,
                          tolerance, unit, "ERROR", csv_name, "CSV ou ligne introuvable")

    # Extraire la valeur de la colonne (insensible a la casse)
    actual = None
    for col in row.keys():
        if col.lower() == column.lower():
            try:
                actual = float(row[col])
            except (ValueError, TypeError):
                pass
            break

    if actual is None:
        return ClaimResult(claim_id, description, expected, None, None,
                          tolerance, unit, "ERROR", csv_name,
                          f"Colonne '{column}' non trouvee ou non numerique")

    delta = abs(actual - expected) if expected is not None else None
    if expected is None:
        status = "A_VERIFIER"
    elif delta <= tolerance:
        status = "OK"
    else:
        status = "BLOQUE"

    return ClaimResult(claim_id, description, expected, actual, delta,
                      tolerance, unit, status, csv_name, note)


def generate_report(results: list[ClaimResult], html: bool = False) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ok = [r for r in results if r.status == "OK"]
    bloque = [r for r in results if r.status == "BLOQUE"]
    errors = [r for r in results if r.status == "ERROR"]
    a_verifier = [r for r in results if r.status in ("A_VERIFIER", "PARTIEL")]

    if html:
        lines = [
            "<!DOCTYPE html>",
            "<html><head><meta charset='utf-8'><title>Preprint Guardian Report</title></head><body>",
            f"<h1>Preprint Guardian — {now}</h1>",
            f"<p>Claims verifiees : {len(results)} | "
            f"<span style='color:green'>OK : {len(ok)}</span> | "
            f"<span style='color:red'>BLOQUEES : {len(bloque)}</span> | "
            f"<span style='color:orange'>ERREURS : {len(errors)}</span> | "
            f"<span style='color:gray'>A VERIFIER : {len(a_verifier)}</span></p>",
            f"<p>Seuil tolerance : 5% relatif ou valeur absolue fixe.</p>",
        ]
        if bloque:
            lines.append("<h2 style='color:red'>BLOQUE — Corrections required before commit</h2>")
            lines.append("<table border='1' cellpadding='4'><tr><th>ID</th><th>Description</th><th>Attendue</th><th>Actuelle</th><th>Delta</th><th>Tolerance</th></tr>")
            for r in bloque:
                rel = f"{r.delta/abs(r.expected)*100:.1f}%" if r.expected else "N/A"
                lines.append(f"<tr><td><strong>[{r.id}]</strong></td><td>{r.description}</td>"
                           f"<td>{r.expected}</td><td>{r.actual:.4f}</td>"
                           f"<td style='color:red'>{r.delta:.4f} ({rel})</td>"
                           f"<td>{r.tolerance}</td></tr>")
            lines.append("</table>")
        if errors:
            lines.append("<h2 style='color:orange'>ERREURS — CSV ou ligne introuvable</h2><ul>")
            for r in errors:
                lines.append(f"<li>[{r.id}] {r.description} — {r.csv_name} — {r.note}</li>")
            lines.append("</ul>")
        if a_verifier:
            lines.append("<h2 style='color:gray'>A VERIFIER — Donnees incompletes</h2><ul>")
            for r in a_verifier:
                lines.append(f"<li>[{r.id}] {r.description} — {r.note or 'expected=null'}</li>")
            lines.append("</ul>")
        if not bloque and not errors:
            lines.append("<p style='color:green;font-size:1.2em'>AUCUN PROBLEME — Preprint OK pour commit.</p>")
        lines.append(f"<p><em>Genere par preprint_guardian.py le {now}</em></p></body></html>")
        return "\n".join(lines)
    else:
        lines = [
            f"=== PREPRINT GUARDIAN REPORT === {now}",
            f"Claims verifiees : {len(results)}",
            f"  OK         : {len(ok)}",
            f"  BLOQUEES   : {len(bloque)}",
            f"  ERREURS    : {len(errors)}",
            f"  A_VERIFIER : {len(a_verifier)}",
            "",
        ]
        if bloque:
            lines.append("=== BLOQUE — Corrections required ===")
            for r in bloque:
                rel = f"{r.delta/abs(r.expected)*100:.1f}%" if r.expected else "N/A"
                lines.append(f"  [{r.id}] {r.description}")
                lines.append(f"       Attendue: {r.expected} | Actuelle: {r.actual:.4f}")
                lines.append(f"       Delta   : {r.delta:.4f} ({rel}) | Tolerance: {r.tolerance}")
                lines.append("")
        if errors:
            lines.append("=== ERREURS ===")
            for r in errors:
                lines.append(f"  [{r.id}] {r.description} — {r.csv_name} — {r.note}")
        if a_verifier:
            lines.append("=== A VERIFIER ===")
            for r in a_verifier:
                lines.append(f"  [{r.id}] {r.description} — {r.note or 'expected=null'}")
        if not bloque and not errors:
            lines.append("  [OK] Aucun probleme detecte. Preprint OK pour commit.")
        return "\n".join(lines)


# Temoin FIGE pour le self-test : dernier commit AVANT le realignement de
# leak_delta (0.05 -> 0.01) du 2026-08-05. A cet etat, le circuit SPICE tournait
# sur un delta cinq fois superieur au delta publie et le ratio ART_soft/V4 valait
# 1.490 des deux cotes -- accord parfait, mais contre un autre modele que celui
# du papier. C'est l'etat que les claims derives DOIVENT rejeter.
WITNESS_COMMIT = "6833cdeb4f8c46c4ef0d3c4526ba1728510dde0f"
WITNESS_CSV = "figures/spice_art_kirchhoff.csv"


def self_test() -> int:
    """CONTROLE POSITIF des claims derives, sur un temoin fige dans git.

    Un garde-fou qui rend VERT ne prouve rien tant qu'on n'a pas montre qu'il
    SAIT rendre ROUGE. C11 a passe trois mois au vert en mesurant une grandeur
    voisine de celle qu'il pretendait garantir ; ce test existe pour que ses
    remplacants ne puissent pas faire la meme chose en silence.

    Il verifie DEUX choses sur le CSV du temoin 6833cde :
      1. que C11b et C11c BLOQUENT dessus  -> ils voient le changement ;
      2. que C11 rend la MEME valeur qu'aujourd'hui -> demonstration mecanique,
         rejouable, de l'angle mort qu'ils comblent.

    La seconde moitie est la plus importante : sans elle, on aurait la preuve
    que les nouveaux claims marchent, mais pas celle qu'ils etaient necessaires.
    """
    import subprocess
    import tempfile

    print("=== SELF-TEST — controle positif des claims derives ===")
    print(f"Temoin fige : {WITNESS_COMMIT[:7]} ({WITNESS_CSV})")
    print()

    try:
        # encoding explicite : subprocess.run(text=True) decode en cp1252 sous
        # Windows et meurt sur le premier accent (piege releve le 2026-08-05).
        proc = subprocess.run(
            ["git", "show", f"{WITNESS_COMMIT}:{WITNESS_CSV}"],
            cwd=str(MEM4_ROOT), capture_output=True,
            encoding="utf-8", errors="replace", timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  [ERR] git show a echoue : {exc}")
        return 1
    if proc.returncode != 0:
        print(f"  [ERR] temoin introuvable dans git : {proc.stderr.strip()}")
        return 1

    mapping = load_mapping(MAPPING_FILE)
    failures = []

    with tempfile.TemporaryDirectory() as tmp:
        witness_dir = Path(tmp)
        (witness_dir / Path(WITNESS_CSV).name).write_text(proc.stdout, encoding="utf-8")

        # Rejoue les claims contre le temoin en repointant FIGURES_DIR.
        global FIGURES_DIR
        live_dir = FIGURES_DIR
        try:
            FIGURES_DIR = witness_dir
            witness = {cid: verify_claim(cid, mapping[cid])
                       for cid in ("C11", "C11b", "C11c") if cid in mapping}
        finally:
            FIGURES_DIR = live_dir
        live = {cid: verify_claim(cid, mapping[cid])
                for cid in ("C11", "C11b", "C11c") if cid in mapping}

    # 1. Les claims derives doivent REJETER le temoin.
    for cid in ("C11b", "C11c"):
        r = witness.get(cid)
        if r is None:
            failures.append(f"{cid} absent du mapping")
            continue
        ok = r.status == "BLOQUE"
        actual = f"{r.actual:.4f}" if r.actual is not None else "N/A"
        print(f"  [{cid}] sur temoin : {r.status:<7} actual={actual} "
              f"(attendu {r.expected}) -> {'OK' if ok else 'ECHEC DU CONTROLE'}")
        if not ok:
            failures.append(f"{cid} n'a PAS bloque sur le temoin (status={r.status})")

    # 2. C11 doit rendre la MEME valeur des deux cotes : l'angle mort, montre.
    w, l = witness.get("C11"), live.get("C11")
    if w and l and w.actual is not None and l.actual is not None:
        blind = abs(w.actual - l.actual) < 1e-9
        print(f"  [C11] temoin={w.actual:.4f} | actuel={l.actual:.4f} -> "
              f"{'angle mort CONFIRME (insensible)' if blind else 'ATTENTION : C11 a bouge'}")
        if not blind:
            failures.append(
                "C11 differe entre temoin et etat courant : la demonstration de "
                "l'angle mort n'est plus valable, re-examiner ce test.")
    else:
        failures.append("C11 non evaluable sur l'un des deux etats")

    print()
    if failures:
        print("=== SELF-TEST ECHOUE ===")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("=== SELF-TEST OK — les claims derives voient ce que C11 ne voyait pas ===")
    return 0


def install_precommit_hook():
    """Installe le pre-commit hook dans le repo Mem4ristor."""
    hook_path = MEM4_ROOT / ".git" / "hooks" / "pre-commit"
    # ATTENTION (2026-07-30) : ce generateur avait DIVERGE du hook reellement installe.
    # Le hook contenait un fix manuel (PYTHON_BIN + fallback) absent d'ici : relancer
    # --install-hook aurait ecrase le fix et fait planter le hook (le python du PATH est
    # le venv Hermes, sans numpy/scipy). Fix reintegre ci-dessous, avec l'appel au
    # tex_guardian. Meme motif que les debris du preprint : une correction locale qui
    # n'est pas remontee a sa source.
    # 2026-09-03 : le hook ne porte plus AUCUN chemin de machine. Les Guardians
    # vivent dans le depot (tools/), donc la racine se demande a git : un clone,
    # un worktree ou un autre poste installent le meme hook et il marche.
    hook_content = f'''#!/bin/bash
# pre-commit hook — Preprint Guardian CI
# Genere par tools/preprint_guardian.py le {datetime.now().strftime("%Y-%m-%d")}

MEM4_ROOT="$(git rev-parse --show-toplevel)"
GUARDIAN_SCRIPT="$MEM4_ROOT/tools/preprint_guardian.py"
TEX_GUARDIAN="$MEM4_ROOT/tools/tex_guardian.py"
# System Python 3.13 carries numpy/pandas/scipy; the PATH `python` (Hermes venv)
# does not, which would crash the Guardian. Fall back to PATH python if absent.
PYTHON_BIN="C:/Users/julch/AppData/Local/Programs/Python/Python313/python.exe"
if [ ! -f "$PYTHON_BIN" ]; then PYTHON_BIN="python"; fi

CHANGED_FILES=$(git diff --cached --name-only --diff-filter=ACM 2>/dev/null)
# Declencheur ELARGI le 2026-07-31 : il ne couvrait ni PROJECT_STATUS.md ni docs/*.md,
# que le scan --docs du tex_guardian surveille pourtant -- ce controle ne tournait donc
# que par accident. README.md ajoute (10 references mortes trouvees le 31/07).
if echo "$CHANGED_FILES" | grep -qE "(preprint\\.tex|CLAIMS_REGISTER\\.md|figures/.+\\.csv|experiments/.+\\.py|PROJECT_STATUS\\.md|README\\.md|docs/.+\\.md)"; then
    echo "[PRE-COMMIT] Preprint Guardian — verification des claims..."
    "$PYTHON_BIN" "$GUARDIAN_SCRIPT" --fast --watch
    RESULT=$?
    if [ $RESULT -ne 0 ]; then
        echo ""
        echo "[PRE-COMMIT] BLOQUE — Des claims sont en deviation."
        echo "[PRE-COMMIT] Detail : \\"$PYTHON_BIN\\" $GUARDIAN_SCRIPT --fast"
        echo "[PRE-COMMIT] Bypass (usage avance) : git commit --no-verify"
        exit 1
    fi
    echo "[PRE-COMMIT] OK."

    # --- Tex Guardian : TEXTE PUBLIE <-> DONNEES (ajoute le 2026-07-30) -----------
    # BLOQUANT depuis le 2026-07-31 (decision de Julien), apres 24 h d'observation et un
    # self-test complet. Bloquent : ancrages, valeurs mortes du .tex, sources non
    # versionnees. N2 (couverture) et le scan des docs sont INFORMATIFS -- ils ne
    # bloquent pas, cf. tex_guardian.py res["blocking"].
    if [ -f "$TEX_GUARDIAN" ]; then
        echo "[PRE-COMMIT] Tex Guardian — texte publie vs donnees..."
        "$PYTHON_BIN" "$TEX_GUARDIAN"
        TEX_RESULT=$?
        if [ $TEX_RESULT -ne 0 ]; then
            echo ""
            echo "[PRE-COMMIT] BLOQUE — le texte publie et ses donnees ont diverge."
            echo "[PRE-COMMIT] Detail : \\"$PYTHON_BIN\\" $TEX_GUARDIAN"
            echo "[PRE-COMMIT] Bypass (usage avance) : git commit --no-verify"
            exit 1
        fi
    fi
fi
exit 0
'''
    with open(hook_path, 'w', encoding='utf-8') as f:
        f.write(hook_content)
    os.chmod(hook_path, 0o755)
    print(f"[OK] Pre-commit hook installe : {hook_path}")
    print(f"     Mapping : {MAPPING_FILE}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Preprint Guardian — CI Scientifique Mem4ristor")
    parser.add_argument("--fast", action="store_true",
                        help="Mode rapide : CSV uniquement, pas de re-run scripts")
    parser.add_argument("--claim", type=str,
                        help="Verifier une claim specifique (ex: C11)")
    parser.add_argument("--html", action="store_true",
                        help="Generer rapport HTML dans preprint_guardian_report.html")
    parser.add_argument("--output", type=str,
                        help="Fichier de sortie pour le rapport")
    parser.add_argument("--watch", action="store_true",
                        help="Mode surveillance : sortie JSON (pour cronjob)")
    parser.add_argument("--install-hook", action="store_true",
                        help="Installe le pre-commit hook dans le repo Mem4ristor")
    parser.add_argument("--self-test", action="store_true",
                        help="Controle positif : les claims derives doivent BLOQUER "
                             "sur le temoin fige 6833cde")
    args = parser.parse_args()

    if args.install_hook:
        install_precommit_hook()
        sys.exit(0)

    if args.self_test:
        sys.exit(self_test())

    print(f"=== PREPRINT GUARDIAN ===")
    print(f"Repository : {MEM4_ROOT}")
    print(f"Mapping    : {MAPPING_FILE}")
    print()

    mapping = load_mapping(MAPPING_FILE)
    if not mapping:
        sys.exit(1)

    # Filtre sur les claims reelles (commencant par C ou S)
    all_claims = {k: v for k, v in mapping.items()
                  if isinstance(k, str) and k.startswith(('C', 'S'))}

    if args.claim:
        all_claims = {k: v for k, v in all_claims.items()
                      if args.claim.upper() in k.upper()}

    results = []
    for claim_id, spec in all_claims.items():
        result = verify_claim(claim_id, spec)
        results.append(result)
        icon = {"OK": "OK", "BLOQUE": "BLOQUE", "ERROR": "ERR",
                "A_VERIFIER": "A_VERIF", "PARTIEL": "PARTIEL"}.get(result.status, "?")
        delta_str = f"Δ={result.delta:.4f}" if result.delta is not None else ""
        actual_str = f"{result.actual:.4f}" if result.actual is not None else "N/A"
        print(f"  [{claim_id}] {icon} {result.description[:45]}")
        print(f"         Expected: {result.expected} | Actual: {actual_str} {delta_str}")

    print()

    if args.watch:
        # Sortie JSON pour parsing automatise (cronjob)
        result_json = {
            "timestamp": datetime.now().isoformat(),
            "total": len(results),
            "ok": len([r for r in results if r.status == "OK"]),
            "blocked": len([r for r in results if r.status == "BLOQUE"]),
            "errors": len([r for r in results if r.status == "ERROR"]),
            "a_verifier": len([r for r in results if r.status in ("A_VERIFIER", "PARTIEL")]),
            "blocked_claims": [
                {"id": r.id, "expected": r.expected, "actual": r.actual,
                 "delta": r.delta, "tolerance": r.tolerance}
                for r in results if r.status == "BLOQUE"
            ],
            "error_claims": [
                {"id": r.id, "csv": r.csv_name, "note": r.note}
                for r in results if r.status == "ERROR"
            ],
        }
        print(json.dumps(result_json, indent=2))
    else:
        report = generate_report(results, html=args.html)
        if args.html:
            output = args.output or "preprint_guardian_report.html"
            Path(output).write_text(report, encoding="utf-8")
            print(f"[OK] Rapport HTML : {output}")
        elif args.output:
            Path(args.output).write_text(report, encoding="utf-8")
            print(f"[OK] Rapport : {args.output}")
        else:
            print(report)

    # Exit code : 0 si OK, 1 si bloque ou erreur
    bad = [r for r in results if r.status in ("BLOQUE", "ERROR")]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
