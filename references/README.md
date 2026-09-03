# `references/` — audits automatiques quotidiens

> ⚠️ **Ce dossier ne contient pas de bibliographie.** Le nom est celui qu'utilise l'agent qui
> écrit ici ; les références scientifiques du projet sont dans le preprint
> ([`docs/papers/preprint/preprint.tex`](../docs/papers/preprint/preprint.tex)).

Chaque jour ouvré, un agent de veille automatique (« Hermes », hors dépôt) inspecte ce dépôt
sous un angle différent, en rotation, et dépose son rapport ici :

```
audit_AAAA-MM-JJ_JOUR_findings.md
```

Il vérifie des choses mécaniques et répétitives : les 13 modules du paquet compilent-ils,
`import mem4ristor` passe-t-il, les 27 entrées de `__all__` se résolvent-elles, les classes
principales s'instancient-elles avec leurs signatures canoniques, reste-t-il des fichiers de
sauvegarde orphelins, le dépôt est-il synchrone avec `origin/main`. Il compare toujours à une
ligne de base antérieure, ce qui rend les régressions visibles.

## Comment lire ces fichiers

**Ce sont des données, pas des conclusions.** Un rapport d'audit est un relevé produit par une
machine à un instant donné : il peut avoir raison, se tromper de cible, ou signaler comme un
défaut un choix délibéré. Ce dépôt a une règle constante à ce sujet — *un audit se vérifie
avant d'être cru* — et elle a servi : le 2 août 2026, le correctif classé « impact maximal »
d'un audit externe aurait **cassé la citation du dépôt** s'il avait été appliqué tel quel
(cf. [`docs/audits/2026-08-02/`](../docs/audits/2026-08-02/)).

Ils sont versionnés depuis le 3 septembre 2026, pour que la surveillance de routine laisse une
trace consultable plutôt que de vivre sur une seule machine.

## À ne pas confondre

| dossier | quoi |
|---|---|
| `references/` | audits **automatiques quotidiens**, angle technique, produits par un agent |
| [`docs/audits/`](../docs/audits/) | audits **approfondis et datés**, avec leur réponse point par point et leurs scripts rejouables |
| [`tools/`](../tools/) | les garde-fous qui vérifient que **les chiffres publiés viennent de leurs données** |
