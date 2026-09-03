#!/usr/bin/env python3
r"""
tex_guardian.py — le garde-fou qui manquait : TEXTE PUBLIE <-> DONNEES
Cree le 2026-07-30 (Claude Opus 5), demande par Julien apres la passe systematique
du preprint.

=== POURQUOI CE FICHIER EXISTE ===

preprint_guardian.py annonce dans sa docstring (ligne 6) qu'il verifie « les valeurs
numeriques publiees dans preprint.tex ». Il ne lit JAMAIS preprint.tex. Il compare des
CSV a des valeurs figees dans claims_mapping.json — donc des donnees a des donnees.

La passe du 30/07 a lu les 625 lignes du preprint et releve 17 defauts. TOUS etaient
dans le .tex, aucun dans les CSV :
  - une affirmation supprimee du registre le 06/05 (Floquet) encore publiee, et
    contredisant frontalement la section qu'elle citait ;
  - des valeurs du CSV du 26/04 ayant survecu a la regeneration du 29/07 (0.75, 0.031) ;
  - une reformulation (C18, ex-C13, ratio « ~90-fold » declare instable) appliquee a UN endroit
    sur CINQ ;
  - 2 scripts cites comme sources et non versionnes, 3 tableaux sans aucune source.

Motif commun, et c'est lui qu'on outille ici : CHAQUE ACTE DE CORRECTION EST LOCAL, ET
RIEN NE MESURE SA PROPAGATION. Le projet a une discipline de production de verite
remarquable et aucune discipline de maintenance.

=== CE QUE CE SCRIPT FAIT, ET CE QU'IL NE FAIT PAS ===

  N1  ANCRAGE (le coeur)   : chaque nombre PUBLIE declare dans tex_anchors.json est
                             compare a la valeur REELLE du CSV (pas a 'expected' du
                             mapping : si le CSV bouge, l'ancre doit casser meme si
                             quelqu'un a mis le mapping a jour).
  N2  COUVERTURE           : une valeur canonique qu'aucune ligne du .tex ne cite ->
                             le claim ne protege rien de publie (informatif).
  N3  VALEURS MORTES       : registre des nombres explicitement REMPLACES (0.031, 0.751,
                             tau_u=12.5, le « +985% »...). Un chiffre mort ne doit plus
                             apparaitre nulle part. Exact, sans faux positif.

      /!\ RESULTAT NEGATIF CONSERVE : N3 a d'abord ete concu comme un filet par PROXIMITE
      numerique (tout nombre proche d'une valeur canonique). Mesure sur le preprint :
      603 suspects, quasi tous absurdes (« 3.14 proche de 3.1234 »). Ce n'etait pas une
      fenetre a regler — pour attraper le cas cible (0.75 vs 0.696680 = 7.6% d'ecart) il
      faut une bande >= 8%, ou il reste des centaines de faux positifs. Sans lien
      SEMANTIQUE entre un nombre et la quantite qu'il denote, la proximite ne dit rien.
      Retire, pas bricole.

Ce qu'il NE fait PAS, et il faut le savoir pour ne pas s'en croire protege :
  - il ne detecte aucun defaut de PROSE (le « ~90-fold », le tau_u inverse, « five
    mechanisms ») : ce sont des mots, pas des nombres ;
  - il ne detecte pas un chiffre orphelin eloigne de toute valeur canonique, SAUF s'il
    est explicitement ancre (c'est le cas du 0.031 de tab:benchmarks, ancre expres) ;
  - il ne verifie pas qu'un script cite existe ou soit versionne (--sources le fait, et
    depuis le 05/08 il regarde AUSSI les sources citees comme dossier — voir
    cited_sources()) ;
  - il ne verifie RIEN sur les fichiers cites par la doc du depot (README,
    REPRODUCE_RESULTS, CLAIMS_REGISTER) : le 05/08 y a mesure 17 references mortes vers
    experiments/scratch/*.py. C'est docs/audits/2026-08-05/check_dead_refs.py qui les
    compte, et il n'est pas branche sur ce garde-fou.

=== CRITERES DE REUSSITE, ECRITS AVANT LE CODE (verifies apres, cf. --self-test) ===

  G1  Sur la version HEAD du .tex (avant les corrections du 30/07), l'ancre
      'alpha-sweep-vs-frozen' DOIT echouer (le texte disait 0.75).
  G2  Sur la version corrigee, elle DOIT passer.
  G3  'tab-benchmarks-sync-this-work' DOIT rester TEX_STALE dans les deux versions
      (0.031 n'est pas un arrondi de 0.002312) — l'ancre est rouge exprès.
  G4  TEST DE POUVOIR : muter une valeur JUSTE du .tex doit produire TEX_STALE. Sans ce
      test, un rapport « tout va bien » ne prouve rien. (Lecon du 27/07 : un controle qui
      donne le bon chiffre pour la mauvaise raison est un controle en panne.)
  G5  Zero alerte sur les 8 valeurs verifiees a la main le 30/07 et justes.
  G6  Le registre des valeurs mortes : DOIT trouver le 0.031 sur le temoin figé et NE
      DOIT RIEN trouver sur l'actuel. (Ajoute avec N3 ; jamais inscrit ici — repare le
      2026-08-05, l'omission etant exactement ce que cet outil surveille.)
  G7  Les sources citees comme DOSSIER (ajoute le 2026-08-05). Trois exigences a la fois :
      la regle doit EXTRAIRE des dossiers (pas zero), les signaler sur le temoin 7f481d7
      — ou experiments/lambda2_foundation_20260701/, cite par preprint.tex:351 pour la
      refutation de lambda2, n'etait pas versionne — et se taire sur l'actuel.

Usage :
  python tex_guardian.py                      # verification complete
  python tex_guardian.py --tex-file X.tex     # verifier une AUTRE version (ex: HEAD)
  python tex_guardian.py --sources            # audit des scripts/CSV cites : existent ? versionnes ?
  python tex_guardian.py --self-test          # G1..G7 : prouve que le detecteur detecte
  python tex_guardian.py --json               # sortie machine (hook, cron)
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).parent))
from preprint_guardian import load_mapping, verify_claim, MEM4_ROOT, MAPPING_FILE  # noqa: E402

ANCHORS_FILE = Path(__file__).parent / "tex_anchors.json"

# Un nombre decimal du .tex. On exige des decimales : les entiers produisent trop de
# bruit (tailles de reseau, numeros de section, comptes de graines).
NUM_RE = re.compile(r"(?<![\d.])(\d+\.\d+)(?![\d])")
# Lignes a ignorer : commentaires LaTeX purs, et tout ce qui ressemble a une date/version.
DATE_RE = re.compile(r"(20\d\d[-/.]\d|v\d+\.\d+\.\d+|arXiv:\d)")


def is_valid_rounding(text_value: str, csv_value: float) -> bool:
    """La chaine du .tex est-elle une ecriture legitime (arrondie ou tronquee) de la
    valeur du CSV ? Tolerance = une unite du dernier chiffre significatif ecrit, ce qui
    couvre arrondi au plus proche ET troncature sans laisser passer une vraie derive."""
    try:
        t = float(text_value)
    except ValueError:
        return False
    decimals = len(text_value.split(".")[1]) if "." in text_value else 0
    return abs(t - csv_value) <= 10.0 ** (-decimals) + 1e-12


def csv_values(mapping: dict) -> dict:
    """Valeur REELLE lue dans chaque CSV, via le Guardian existant (pas de duplication
    de la logique de filtrage, donc pas de divergence possible entre les deux outils)."""
    out = {}
    for cid, spec in mapping.items():
        if not (isinstance(cid, str) and cid.startswith(("C", "S"))):
            continue
        r = verify_claim(cid, spec)
        if r.actual is not None:
            out[cid] = {"value": r.actual, "csv": r.csv_name,
                        "desc": r.description, "status": r.status}
    return out


def read_tex(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def check_anchors(lines: list[str], anchors: list[dict], cvals: dict) -> list[dict]:
    findings = []
    for a in anchors:
        aid, val, claim = a["id"], a["value"], a["claim"]
        ctx = a.get("context")
        hit_line = None
        for i, line in enumerate(lines, 1):
            if val in line and (ctx is None or ctx in line):
                hit_line = i
                break

        if hit_line is None:
            findings.append({"level": "ANCHOR_LOST", "anchor": aid, "claim": claim,
                             "line": None, "tex": val,
                             "msg": f"chaine '{val}'" + (f" + contexte '{ctx}'" if ctx else "")
                                    + " absente du .tex : le texte a change sous l'ancre"})
            continue

        if claim not in cvals:
            findings.append({"level": "NO_SOURCE", "anchor": aid, "claim": claim,
                             "line": hit_line, "tex": val,
                             "msg": f"le claim {claim} ne rend aucune valeur (CSV absent ?)"})
            continue

        cv = cvals[claim]["value"]
        if is_valid_rounding(val, cv):
            findings.append({"level": "OK", "anchor": aid, "claim": claim,
                             "line": hit_line, "tex": val, "csv": cv,
                             "msg": f"{val} est une ecriture valide de {cv:.6g} ({cvals[claim]['csv']})"})
        else:
            findings.append({"level": "TEX_STALE", "anchor": aid, "claim": claim,
                             "line": hit_line, "tex": val, "csv": cv,
                             "msg": f"le .tex publie {val}, le CSV dit {cv:.6g} "
                                    f"({cvals[claim]['csv']}) — texte non synchronise"})
    return findings


def check_coverage(lines: list[str], cvals: dict, anchors: list[dict]) -> list[dict]:
    """N2 — un claim non ancre protege-t-il quelque chose de PUBLIE ?

    ⚠️ DEUX LIMITES MESUREES, qui vont en sens OPPOSES. N2 est INFORMATIF et non
    bloquant (cf. res["blocking"]), donc aucune ne bloque rien a tort — mais la
    seconde RASSURE a tort, ce qui est pire qu'une absence de controle.

    1. FAUX SIGNALEMENT (connu, structurel, 31/07). NUM_RE exige `\\d+\\.\\d+` :
       les entiers nus sont invisibles. C18 est donc affiche UNCITED alors qu'il
       backe « Cohen's d ≈ 9 », publie TROIS fois — le .tex l'ecrit en entier.
       Ne jamais retirer un claim sur la seule foi d'un UNCITED : le 31/07, le
       faire aurait desarme le garde-fou du resultat central.

    2. FAUX SILENCE (mesure le 06/08, NOUVEAU). is_valid_rounding tolere une
       unite du dernier chiffre ecrit, en ABSOLU et non en relatif. Pour un claim
       de petite magnitude, n'importe quel nombre voisin du texte le « couvre » :
       mesure sur le preprint, C11b (ratio SPICE = 1.1338) est valide 8 fois par
       un simple `1.2` present ailleurs, et C11 (0.0) l'est 52 fois par des
       `0.00`/`0.01`/`0.1`. Or le .tex ne contient AUCUNE valeur SPICE — verifie.
       Donc : l'ABSENCE d'UNCITED ne prouve PAS qu'un claim protege du publie.
       C'est le meme motif que les trois angles morts du 05/08 — un controle qui
       repond a cote et reste vert — et c'est en cherchant pourquoi N2 ne
       signalait pas C11b qu'il a ete trouve.

       CAUSE EXACTE (precisee le 06/08 apres mesure). Ce n'est pas « absolu au
       lieu de relatif » : is_valid_rounding est ecrite pour N1 — verifier qu'UN
       nombre DESIGNE du texte est l'ecriture d'un CSV — et sa tolerance d'une
       unite y est prudente a dessein (elle couvre la troncature). N2 reutilise
       la meme fonction pour l'usage OPPOSE : balayer 802 lignes en acceptant
       n'importe quel nombre. La meme prudence y devient du bruit. Mesure :
       290 hits, dont 135 disparaissent si l'on exige l'arrondi au plus proche.
       AMPLEUR REELLE : sur les 10 claims soumis a N2, C11b est le SEUL verdict
       faux — le defaut est etroit, mais il rend le verdict NON CONCLUANT partout.

    ⚠️ LA CORRECTION EVIDENTE EST PIEGEE, et c'est pourquoi elle n'est pas faite
    ici. Resserrer a 0.5 unite corrigerait C11b, mais la fonction est PARTAGEE
    avec check_anchors (N1), qui est BLOQUANT : l'ancre parfaitement legitime de
    C06 — le .tex ecrit « \\approx 0.296 » pour un CSV a 0.295477, ecart 0.000523,
    et l'ancre elle-meme note « CSV: 0.2955 » — basculerait en TEX_STALE et
    casserait les commits. La sortie propre est de DECOUPLER les deux seuils :
    strict pour N2, inchange pour N1. Chantier a part, DECISION DE JULIEN.
    """
    body = "\n".join(lines)
    ancres_par_claim = {a["claim"] for a in anchors}
    out = []
    for cid, info in cvals.items():
        if cid in ancres_par_claim:
            continue
        cv = info["value"]
        cite = any(is_valid_rounding(m.group(1), cv) for m in NUM_RE.finditer(body))
        if not cite:
            out.append({"level": "UNCITED", "claim": cid, "csv_value": cv,
                        "msg": f"aucune ligne du .tex ne cite {cv:.6g} ({info['csv']}) — "
                               f"ce claim ne protege rien de publie, ou le texte cite autre chose"})
    return out


def scan_stale_values(lines: list[str], perimees: list[dict]) -> list[dict]:
    """Registre des VALEURS MORTES : un nombre explicitement remplace ne doit plus
    apparaitre. Exact, sans faux positif — contrairement a la detection par proximite
    numerique essayee puis retiree le 30/07 (603 suspects, cf. _n3_retire).

    C'est le pendant numerique du registre des claims supprimes de
    docs/CLAIMS_REGISTER.md : ce registre-la contenait Floquet depuis le 06/05, et
    personne ne le relisait. Ici, la relecture est mecanique."""
    out = []
    for p in perimees:
        val = p["value"]
        garde = p.get("sauf_si_ligne_contient")
        # Contrainte semantique : pour les valeurs courtes (0.75, 985...), l'ecriture seule
        # ne suffit pas a identifier la quantite. Faux positif verifie le 30/07 : '-0.75' en
        # L353 est un coefficient de Pearson, pas la synchronie du doute gele.
        besoin = p.get("contexte_requis")
        # Frontieres : on ne veut pas que '0.75' matche dans '0.751' ou '12.5' dans '12.58'.
        pat = re.compile(r"(?<![\d.])" + re.escape(val) + r"(?![\d])")
        for i, line in enumerate(lines, 1):
            # PAS de filtre DATE_RE ici — bug attrape le 30/07 par relecture : L182 contient
            # « (audit 2026-04-22) », donc le filtre anti-date sautait la ligne ET le 0.031
            # mort qu'elle publie. DATE_RE servait au scan par proximite (N3, retire) qui
            # ramassait TOUS les nombres ; sur un registre de valeurs exactes il fait rater
            # de vraies trouvailles. Les frontieres de mot suffisent ('985' ne matche pas
            # dans '1985'), doublees d'un contexte_requis sur les valeurs courtes.
            if line.lstrip().startswith("%"):
                continue
            if garde and garde in line:
                continue
            if besoin and besoin.lower() not in line.lower():
                continue
            if pat.search(line):
                out.append({"level": "STALE_VALUE", "claim": p.get("claim"), "line": i,
                            "tex": val, "remplacee_par": p.get("remplacee_par"),
                            "msg": f"L{i}: valeur MORTE '{val}' (remplacee par "
                                   f"'{p.get('remplacee_par')}' le {p.get('quand')}). "
                                   f"{p.get('motif', '')}"})
    return out


def scan_docs(perimees: list[dict], globs: list[str]) -> list[dict]:
    """Le meme registre de valeurs mortes, applique aux DOCUMENTS du depot.

    Motive par la premiere application de la regle de cloture soustractive (30/07) :
    docs/FUTURE_WORK.md:846 citait « 0.031 contre 0.751 » — les deux valeurs mortes cote
    a cote, dans un texte que la regeneration du 29/07 n'avait pas mis a jour. Le Tex
    Guardian ne les voyait pas : il ne lit que le .tex.

    NON BLOQUANT a dessein. Un document d'histoire cite legitimement une valeur morte
    pour dire qu'elle est morte — le registre lui-meme en est plein. Les citations
    justifiees s'acquittent par 'sauf_si_ligne_contient'."""
    out = []
    for g in globs:
        for path in sorted(MEM4_ROOT.glob(g)):
            if not path.is_file():
                continue
            try:
                lignes = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for f in scan_stale_values(lignes, perimees):
                f["level"] = "DOC_STALE_VALUE"
                f["file"] = path.relative_to(MEM4_ROOT).as_posix()
                out.append(f)
    return out


SRC_FILE_RE = re.compile(r"(?:experiments|figures)/[A-Za-z0-9_/]+\.(?:py|csv)")
SRC_DIR_RE = re.compile(r"(?:experiments|figures)/[A-Za-z0-9_/]*/")


def cited_sources(body: str) -> list[str]:
    """Les chemins cites par le texte : fichiers .py/.csv ET dossiers.

    Ajout du 2026-08-05 — LE TROU QUE CETTE FONCTION FERME. Jusqu'ici l'extraction
    exigeait une extension .py ou .csv. Une source citee comme DOSSIER etait donc
    INVISIBLE, et le rapport annoncait « 12/12 sources versionnees » sans mentir
    tout en ne regardant pas tout : preprint.tex:351 renvoyait a
    experiments/lambda2_foundation_20260701/ — les deux experiences controlees qui
    etablissent que lambda2 n'est PAS causal, resultat central de sec:lambda2 cite
    dans l'abstract — et ce dossier vivait dans experiments/scratch/, gitignore.
    Aucun des trois audits externes (02/08, deux le 05/08) ne l'avait vu non plus.

    Meme lecon que le 31/07 sur les ancres : un « 12/12 » prouve que douze fichiers
    sont versionnes, il ne prouve rien sur les sources qu'on n'extrait pas.

    Un dossier PREFIXE d'un fichier deja capte est ecarte : si le texte cite
    experiments/foo/bar.py, tester bar.py suffit, et remonter experiments/foo/ en
    plus produirait un doublon (tout chemin de fichier contient son dossier)."""
    files = sorted(set(SRC_FILE_RE.findall(body)))
    dirs = sorted({d for d in SRC_DIR_RE.findall(body)
                   if not any(f.startswith(d) for f in files)})
    return files + dirs


def audit_sources(lines: list[str], ref: str | None = None) -> list[dict]:
    """Chaque script/CSV/DOSSIER cite par le preprint existe-t-il, et est-il VERSIONNE ?
    (Le 29/07 a montre que experiments/scratch/ est gitignore : ce qui y vit n'est
    pas regenerable par qui clone le depot.)

    `ref` : si fourni, on interroge l'arbre de ce COMMIT au lieu de l'index courant.
    Sert au controle positif du self-test — prouver que la regle attrape un defaut
    reel et date, et pas seulement qu'elle se tait aujourd'hui."""
    body = "\n".join(lines).replace("\\_", "_").replace("\\", "")
    out = []
    for c in cited_sources(body):
        is_dir = c.endswith("/")
        cmd = (["git", "-C", str(MEM4_ROOT), "ls-tree", "-r", "--name-only", ref, "--", c]
               if ref else ["git", "-C", str(MEM4_ROOT), "ls-files", c])
        try:
            tracked = subprocess.run(cmd, capture_output=True, text=True, timeout=20,
                                     encoding="utf-8", errors="replace").stdout.strip()
        except Exception as e:                                    # pragma: no cover
            out.append({"level": "SOURCE_UNKNOWN", "path": c, "msg": f"git indisponible : {e}"})
            continue
        if tracked:
            out.append({"level": "OK", "path": c,
                        "msg": "versionne" + (" (dossier)" if is_dir else "")})
            continue
        leaf = Path(c.rstrip("/")).name
        roots = [MEM4_ROOT / "experiments", MEM4_ROOT / "figures"]
        found = [p for r in roots for p in r.rglob(leaf)
                 if p.is_dir() == is_dir]
        if found:
            where = found[0].relative_to(MEM4_ROOT).as_posix()
            quoi = "dossier" if is_dir else "fichier"
            out.append({"level": "SOURCE_UNVERSIONED", "path": c,
                        "msg": f"{quoi} cite comme '{c}' mais vit en '{where}' et n'est PAS "
                               f"versionne -> non regenerable par qui clone le depot"})
        else:
            out.append({"level": "SOURCE_MISSING", "path": c,
                        "msg": "cite par le preprint et introuvable dans le depot"})
    return out


def audit_producers(mapping: dict) -> list[dict]:
    """N4 — CHAQUE CSV DE CLAIM EST-IL REGENERABLE ? (ajoute le 2026-07-31)

    L'angle mort que cet audit ferme, et il a coute cher deux fois :
    le Guardian garantit que « le chiffre publie = le chiffre du CSV ». Il ne
    garantit RIEN sur la capacite a REGENERER ce CSV. Or c'est precisement ce qui
    a explose le 12/06 (AUDIT-024) : trois CSV anterieurs a mai ne se
    reproduisaient plus du tout — le bruit avait change le 01/05 et personne ne
    pouvait le voir, faute de pouvoir rejouer.

    Mesure du 31/07 qui a motive cet ajout : SEPT claims sur dix-huit (C05, C08,
    C08b, C09, C10, C11, C12) avaient leur unique producteur dans
    experiments/scratch/, gitignore — dont C05, la frontiere lambda2 = 2.31 citee
    dans l'abstract. Le registre s'ouvre pourtant sur « Zero valeur sans script
    reproductible ».

    HEURISTIQUE ET SA LIMITE, ecrites ensemble : on considere qu'un script
    PRODUIT un CSV s'il nomme le fichier ET contient une ecriture (to_csv,
    csv.writer, DictWriter, open(...,'w')). Faux positif possible : un script qui
    nomme le CSV pour le LIRE et ecrit ailleurs. Faux negatif possible : un
    script qui construit le nom par f-string. Ce controle repere donc une
    ABSENCE DE PRODUCTEUR VERSIONNE CONNU, pas une preuve d'irreproductibilite —
    a lire comme une question, jamais comme un verdict (lecon du 28/07 : un
    garde-fou qui tombe n'est pas une alerte tant qu'on n'a pas verifie ce que la
    cible affirme reellement)."""
    ecriture = re.compile(r"to_csv|csv\.writer|DictWriter|open\([^)]*['\"]w")
    try:
        tracked = set(subprocess.run(["git", "-C", str(MEM4_ROOT), "ls-files"],
                                     capture_output=True, text=True,
                                     timeout=30).stdout.split())
    except Exception as e:                                        # pragma: no cover
        return [{"level": "PRODUCER_UNKNOWN", "claim": "-", "msg": f"git indisponible : {e}"}]

    scripts = sorted((MEM4_ROOT / "experiments").rglob("*.py"))
    textes = {}
    for s in scripts:
        try:
            textes[s] = s.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

    out, vus = [], set()
    for cid, info in mapping.items():
        csv_name = info.get("csv") if isinstance(info, dict) else None
        if not csv_name or csv_name in vus:
            continue
        vus.add(csv_name)
        versionnes, orphelins = [], []
        for s, txt in textes.items():
            if csv_name not in txt or not ecriture.search(txt):
                continue
            rel = s.relative_to(MEM4_ROOT).as_posix()
            (versionnes if rel in tracked else orphelins).append(rel)
        if versionnes:
            out.append({"level": "OK", "claim": cid, "csv": csv_name,
                        "msg": f"produit par {versionnes[0]}"})
        elif orphelins:
            out.append({"level": "PRODUCER_UNVERSIONED", "claim": cid, "csv": csv_name,
                        "msg": f"son seul producteur connu ({orphelins[0]}) n'est PAS "
                               f"versionne -> le CSV n'est pas regenerable par qui clone"})
        else:
            out.append({"level": "PRODUCER_MISSING", "claim": cid, "csv": csv_name,
                        "msg": "aucun script du depot ne semble ecrire ce CSV "
                               "(cf. limites de l'heuristique dans la docstring)"})
    return out


BAD = ("TEX_STALE", "ANCHOR_LOST", "NO_SOURCE", "STALE_VALUE",
       "SOURCE_UNVERSIONED", "SOURCE_MISSING")
# PRODUCER_* n'est PAS dans BAD : N4 demarre en OBSERVATION, comme le tex_guardian
# lui-meme du 30 au 31/07. On ne donne pas a un controle neuf le droit de refuser
# un commit avant d'avoir mesure ce qu'il signale — la lecon du 30/07 (N3, 603 faux
# positifs mesures puis retire) a ete payee exactement la-dessus.


def run(tex_path: Path, with_sources: bool = True,
        cvals_override: dict | None = None) -> dict:
    """cvals_override : injecte des valeurs CSV simulees, pour le self-test uniquement.
    Permet de tester le cas « le CSV a bouge, le texte non » SANS toucher aux CSV
    canoniques sur le disque."""
    cfg = json.loads(ANCHORS_FILE.read_text(encoding="utf-8"))
    mapping = load_mapping(MAPPING_FILE)
    cvals = cvals_override if cvals_override is not None else csv_values(mapping)
    lines = read_tex(tex_path)

    res = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tex_file": str(tex_path),
        "tex_lines": len(lines),
        "claims_with_value": len(cvals),
        "anchors": check_anchors(lines, cfg["anchors"], cvals),
        "coverage": check_coverage(lines, cvals, cfg["anchors"]),
        "stale": scan_stale_values(lines, cfg.get("valeurs_perimees", [])),
        "docs": (scan_docs(cfg.get("valeurs_perimees", []), cfg.get("docs_scannes", []))
                 if with_sources else []),
        "sources": audit_sources(lines) if with_sources else [],
        "producers": audit_producers(mapping) if with_sources else [],
    }
    res["blocking"] = [f for grp in ("anchors", "stale", "sources") for f in res[grp]
                       if f["level"] in BAD]
    return res


def print_report(r: dict) -> None:
    print(f"=== TEX GUARDIAN === {r['timestamp']}")
    print(f"Texte   : {r['tex_file']} ({r['tex_lines']} lignes)")
    print(f"Claims  : {r['claims_with_value']} rendent une valeur CSV")
    print()

    ok = [a for a in r["anchors"] if a["level"] == "OK"]
    print(f"--- N1 ANCRAGES ({len(ok)}/{len(r['anchors'])} OK) ---")
    for a in r["anchors"]:
        if a["level"] != "OK":
            loc = f"L{a['line']}" if a.get("line") else "--"
            print(f"  [{a['level']}] {a['anchor']} ({a['claim']}, {loc})")
            print(f"      {a['msg']}")
    if len(ok) == len(r["anchors"]):
        print("  toutes les ancres tiennent.")

    print(f"\n--- N2 COUVERTURE ({len(r['coverage'])} claim(s) non cite(s)) ---")
    for c in r["coverage"] or []:
        print(f"  [{c['level']}] {c['claim']} : {c['msg']}")
    if not r["coverage"]:
        print("  aucun claim non ancre n'est signale.")
    # LIMITE CONNUE, affichee AVEC le verdict et non seulement dans le code
    # (decision de Julien, 06/08) : c'est ici qu'on lit le resultat, donc c'est
    # ici que doit vivre ce qu'il ne prouve pas. Voir check_coverage().
    print("  /!\\ ce verdict n'est pas concluant dans les DEUX sens :")
    print("      - un UNCITED peut etre FAUX (les entiers nus sont invisibles :")
    print("        C18 est signale alors qu'il backe « d = 9 », publie 3 fois) ;")
    print("      - l'ABSENCE d'UNCITED ne prouve RIEN (tolerance heritee de N1 :")
    print("        un nombre voisin sans rapport suffit a « couvrir » un claim).")

    print(f"\n--- N3 VALEURS MORTES ({len(r['stale'])} occurrence(s)) ---")
    for d in r["stale"] or []:
        print(f"  [{d['level']}] {d['msg']}")
    if not r["stale"]:
        print("  aucune valeur perimee ne subsiste dans le texte.")

    if r.get("docs") is not None and r["sources"]:
        print(f"\n--- DOCS DU DEPOT ({len(r['docs'])} valeur(s) morte(s), NON bloquant) ---")
        for d in r["docs"]:
            print(f"  [{d['level']}] {d['file']}:{d['line']} — valeur morte '{d['tex']}' "
                  f"(remplacee par '{d['remplacee_par']}')")
        if not r["docs"]:
            print("  aucune valeur perimee dans les documents scannes.")

    if r["sources"]:
        bad = [s for s in r["sources"] if s["level"] != "OK"]
        print(f"\n--- SOURCES CITEES ({len(r['sources']) - len(bad)}/{len(r['sources'])} versionnees) ---")
        for s in bad:
            print(f"  [{s['level']}] {s['path']}")
            print(f"      {s['msg']}")

    prod = r.get("producers") or []
    if prod:
        ko = [p for p in prod if p["level"] != "OK"]
        print(f"\n--- N4 PRODUCTEURS DES CSV ({len(prod) - len(ko)}/{len(prod)} regenerables, "
              f"NON bloquant) ---")
        for p in ko:
            print(f"  [{p['level']}] {p['claim']} ({p['csv']}) — {p['msg']}")
        if not ko:
            print("  chaque CSV de claim a un producteur versionne.")

    print()
    if r["blocking"]:
        print(f"[BLOQUE] {len(r['blocking'])} probleme(s) bloquant(s).")
    else:
        print("[OK] Texte et donnees synchronises sur tout ce qui est ancre.")


def self_test() -> int:
    """G1..G7 : prouve que le detecteur DETECTE. Un rapport vert sans ce test ne vaut rien."""
    import tempfile
    tex_rel = json.loads(ANCHORS_FILE.read_text(encoding="utf-8"))["tex_file"]
    cur = MEM4_ROOT / tex_rel
    print("=== SELF-TEST (G1..G7) ===\n")
    verdicts = []

    def lvl(res, anchor_id):
        return next((a["level"] for a in res["anchors"] if a["anchor"] == anchor_id), "ABSENT")

    now = run(cur, with_sources=False)

    # --- LE TEMOIN FIGE -------------------------------------------------------------
    # Les criteres ont d'abord ete ecrits contre « HEAD ». Ils se sont mis a ECHOUER des que
    # les defauts ont ete corriges (le 0.031 n'existe plus, donc « le 0.031 doit etre
    # detecte » devient faux). C'est EXACTEMENT le defaut que cet outil surveille, applique
    # a l'outil : un test dont la reference bouge sous lui.
    # Correction : ancrer sur un commit FIXE — l'etat du preprint au soir du 29/07, qui
    # contient les vrais defauts trouves le 30/07. Le self-test prouve ainsi en permanence
    # qu'il AURAIT attrape des defauts reels et dates, quoi qu'il arrive ensuite au depot.
    TEMOIN = "cd7484c"
    tem = subprocess.run(["git", "-C", str(MEM4_ROOT), "show", f"{TEMOIN}:{tex_rel}"],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    old = None
    if tem.returncode == 0:
        with tempfile.NamedTemporaryFile("w", suffix=".tex", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(tem.stdout)
            hp = Path(fh.name)
        old = run(hp, with_sources=False)
        hp.unlink(missing_ok=True)

    # G2 : l'ancre corrigee passe sur la version actuelle
    g2 = lvl(now, "alpha-sweep-vs-frozen") == "OK"
    verdicts.append(("G2", g2, f"alpha-sweep-vs-frozen sur le .tex actuel = {lvl(now,'alpha-sweep-vs-frozen')}"))

    # G1 : la meme ancre DOIT echouer sur le temoin (il y publiait 0.75)
    if old is not None:
        g1 = lvl(old, "alpha-sweep-vs-frozen") in BAD
        verdicts.append(("G1", g1, f"sur le temoin {TEMOIN} = {lvl(old,'alpha-sweep-vs-frozen')} "
                                   f"(attendu : bloquant)"))
    else:
        verdicts.append(("G1", False, f"git show {TEMOIN} a echoue"))

    # G3 : tab:benchmarks — resolu aujourd'hui cote actuel, DETECTE cote temoin.
    # Le 30/07, cette ancre etait volontairement rouge (0.031 publie contre 0.002312 mesure) ;
    # decision de Julien le meme jour : option lattice -> la ligne porte desormais 0.0197
    # (C07b) et H = 4.09 (C01 depuis le 31/07), les deux colonnes venant enfin des memes runs.
    g3_now = lvl(now, "tab-benchmarks-sync-this-work") == "OK"
    g3_old = (old is not None) and lvl(old, "tab-benchmarks-sync-this-work") in BAD
    verdicts.append(("G3", g3_now and g3_old,
                     f"actuel = {lvl(now,'tab-benchmarks-sync-this-work')} (resolu), "
                     f"temoin = {lvl(old,'tab-benchmarks-sync-this-work') if old else 'N/A'} (detecte)"))

    # --- G4 : TEST DE POUVOIR. Le critere initial etait MAL POSE ; conserve a l'affichage,
    # non deplace, et double des deux controles qui auraient du y etre (G4a, G4b).
    # Ecrit le 2026-07-30 : « muter une valeur juste doit produire TEX_STALE ». Faux par
    # construction — changer le nombre dans le .tex fait disparaitre la chaine ancree, donc
    # ce sens-la rend ANCHOR_LOST. TEX_STALE est le sens INVERSE (le CSV bouge, le texte
    # reste), qui est precisement le scenario du 29/07. Je testais le mauvais sens.
    muted = cur.read_text(encoding="utf-8").replace(
        "Frozen $u$ (no doubt dynamics) & $0.697", "Frozen $u$ (no doubt dynamics) & $0.690")
    with tempfile.NamedTemporaryFile("w", suffix=".tex", delete=False, encoding="utf-8") as fh:
        fh.write(muted)
        mp = Path(fh.name)
    mut = run(mp, with_sources=False)
    niveau_mut = lvl(mut, "tab-ablations-sync-frozen")
    verdicts.append(("G4", None,
                     f"MAL POSE (conserve, non deplace) : attendait TEX_STALE, obtient {niveau_mut}. "
                     f"Un changement du .tex ne PEUT pas rendre TEX_STALE — voir G4a/G4b."))

    # G4a : le vrai test de pouvoir cote TEXTE — une valeur publiee falsifiee doit BLOQUER,
    # quel que soit le code de verdict.
    g4a = niveau_mut in BAD
    verdicts.append(("G4a", g4a, f"0.697 -> 0.690 donne {niveau_mut}, "
                                 f"{'bloquant' if g4a else 'NON bloquant'}"))
    mp.unlink(missing_ok=True)

    # G4b : le vrai test de pouvoir cote DONNEES — on simule une regeneration de CSV
    # (le scenario du 29/07) sans toucher aucun fichier, et TEX_STALE doit tomber.
    reels = csv_values(load_mapping(MAPPING_FILE))
    simule = {k: dict(v) for k, v in reels.items()}
    if "C15" in simule:
        simule["C15"]["value"] = 0.751          # la valeur du CSV du 26/04, comme si elle revenait
    regen = run(cur, with_sources=False, cvals_override=simule)
    g4b = lvl(regen, "tab-ablations-sync-frozen") == "TEX_STALE"
    verdicts.append(("G4b", g4b, f"CSV simule a 0.751 alors que le .tex publie 0.697 -> "
                                 f"{lvl(regen,'tab-ablations-sync-frozen')} (attendu TEX_STALE)"))

    # G6 : le registre des valeurs mortes. Deux exigences OPPOSEES, pour qu'il ne puisse
    # etre ni aveugle ni bavard :
    #   - sur le TEMOIN, il doit trouver le 0.031 (qui y etait publie deux fois, L182 et
    #     L523) -> preuve de pouvoir, permanente ;
    #   - sur l'ACTUEL, il ne doit plus rien trouver -> preuve d'absence de bavardage,
    #     notamment sur le '-0.75' de Pearson en L353, qui n'est pas la valeur morte.
    mortes_now = {s["tex"] for s in now["stale"]}
    mortes_old = {s["tex"] for s in old["stale"]} if old else set()
    g6 = (not mortes_now) and ("0.031" in mortes_old)
    verdicts.append(("G6", g6, f"temoin = {sorted(mortes_old) or 'aucune'} (attendu 0.031), "
                               f"actuel = {sorted(mortes_now) or 'aucune'} (attendu aucune)"))

    # G6b : test de POUVOIR du registre — reinjecter la valeur morte corrigee ce matin
    # doit la faire ressortir. Sinon le registre ne prouve rien.
    reinj = cur.read_text(encoding="utf-8").replace(
        "versus $0.697$ for the frozen-doubt", "versus $0.75$ for the frozen-doubt")
    with tempfile.NamedTemporaryFile("w", suffix=".tex", delete=False, encoding="utf-8") as fh:
        fh.write(reinj)
        rp = Path(fh.name)
    back = run(rp, with_sources=False)
    g6b = any(s["tex"] == "0.75" for s in back["stale"])
    verdicts.append(("G6b", g6b, "0.75 reinjecte en L338 -> "
                                 + ("rattrape" if g6b else "NON rattrape")))
    rp.unlink(missing_ok=True)

    # G7 : LES SOURCES CITEES COMME DOSSIER (ajoute le 2026-08-05, avec la regle qu'il teste).
    # Meme forme que G6 : deux exigences opposees, pour que la regle ne puisse etre ni
    # aveugle ni bavarde. Le temoin est FIGE sur 7f481d7 — le dernier commit AVANT que
    # experiments/lambda2_foundation_20260701/ soit versionne. Ce dossier porte les deux
    # experiences controlees qui refutent la causalite de lambda2, et preprint.tex:351 le
    # citait alors qu'un clone ne l'avait pas : le detecteur doit prouver, en permanence,
    # qu'il aurait attrape CE defaut-la, date et reel.
    TEMOIN_DIR = "7f481d7"
    tex_now = cur.read_text(encoding="utf-8").splitlines()
    src_now = audit_sources(tex_now)
    src_old = audit_sources(tex_now, ref=TEMOIN_DIR)

    def dirs_bad(res):
        return {s["path"] for s in res
                if s["path"].endswith("/") and s["level"] in ("SOURCE_UNVERSIONED", "SOURCE_MISSING")}

    dirs_vus = {s["path"] for s in src_now if s["path"].endswith("/")}
    g7_pouvoir = "experiments/lambda2_foundation_20260701/" in dirs_bad(src_old)
    g7_silence = not dirs_bad(src_now)
    g7_extrait = bool(dirs_vus)          # la regle regarde bien des dossiers, pas zero
    g7 = g7_pouvoir and g7_silence and g7_extrait
    verdicts.append(("G7", g7,
                     f"dossiers extraits = {sorted(dirs_vus) or 'AUCUN'} ; "
                     f"sur le temoin {TEMOIN_DIR} = {sorted(dirs_bad(src_old)) or 'aucun'} "
                     f"(attendu : lambda2_foundation) ; actuel = "
                     f"{sorted(dirs_bad(src_now)) or 'aucun'} (attendu aucun)"))

    # G5 : aucune fausse alerte sur les valeurs verifiees a la main le 30/07
    sains = ["tab-ablations-sync-full", "tab-ablations-sync-frozen", "tab-ablations-lz-full",
             "tab-ablations-lz-frozen", "scaling-h-4x4", "scaling-h-25x25",
             "lambda2-midpoint", "alpha-crit-hopf", "ablations-vs-benchmarks",
             "tab-benchmarks-sync-this-work"]
    faux = [s for s in sains if lvl(now, s) != "OK"]
    g5 = not faux
    verdicts.append(("G5", g5, "aucune fausse alerte" if g5 else f"fausses alertes : {faux}"))

    for name, passed, detail in verdicts:
        tag = "MAL_POSE" if passed is None else ("PASS" if passed else "FAIL")
        print(f"  [{tag}] {name} — {detail}")
    # Un critere MAL_POSE (passed is None) reste AFFICHE en permanence mais ne compte pas
    # comme echec : il est remplace par les controles adjacents, pas repeche.
    echecs = [v[0] for v in verdicts if v[1] is False]
    print()
    print("  TOUS LES CRITERES PASSENT : le detecteur detecte." if not echecs
          else f"  ECHECS : {', '.join(echecs)} — ne pas se fier au rapport vert.")
    return 0 if not echecs else 1


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Tex Guardian — synchronisation texte publie <-> donnees")
    p.add_argument("--tex-file", type=str, help="verifier une autre version du .tex")
    p.add_argument("--sources", action="store_true", help="auditer seulement les sources citees")
    p.add_argument("--self-test", action="store_true", help="prouver que le detecteur detecte (G1..G7)")
    p.add_argument("--json", action="store_true", help="sortie machine")
    p.add_argument("--no-fail", action="store_true", help="toujours sortir 0 (mode observation)")
    args = p.parse_args()

    if args.self_test:
        sys.exit(self_test())

    cfg = json.loads(ANCHORS_FILE.read_text(encoding="utf-8"))
    tex = Path(args.tex_file) if args.tex_file else MEM4_ROOT / cfg["tex_file"]
    if not tex.exists():
        print(f"[ERREUR] .tex introuvable : {tex}")
        sys.exit(2)

    r = run(tex)

    if args.sources:
        for s in r["sources"]:
            print(f"  [{s['level']}] {s['path']} — {s['msg']}")
    elif args.json:
        print(json.dumps(r, indent=2, ensure_ascii=False))
    else:
        print_report(r)

    sys.exit(0 if (args.no_fail or not r["blocking"]) else 1)


if __name__ == "__main__":
    main()
