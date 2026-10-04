# ÉquiAlgo : gouvernance et surveillance

Ce document est la source unique de tout ce que le rapport d'audit et la présentation
disent de la gouvernance. Le prototype de surveillance correspondant est `monitoring.py` :
les indicateurs et les seuils de la section 3 y sont codés tels quels. Les constats sur
lesquels il s'appuie sont établis dans `audit_rapport.ipynb` ; ils ne sont pas rechiffrés
ici.

## 1. Choix assumés

Corriger un biais oblige à trancher des questions qui ne sont pas techniques. Nous disons
ici ce que nous avons choisi, pourquoi, et ce qui revient à l'institution.

**La définition de l'équité : l'égalité des chances, mesurée contre un étalon de mérite
indépendant du comité.** À mérite égal, un candidat d'une région éloignée doit avoir la
même chance d'obtenir la bourse qu'un candidat d'un grand centre. Nous ne retenons pas la
parité comme objectif : une bourse au mérite ne garantit pas un quota par région. Avec
l'étalon retenu (cote R et heures travaillées), les deux groupes comptent la même part de
méritants : le modèle corrigé atteint alors aussi la parité, sans qu'elle ait été imposée
(rapport d'audit, section 6).

**La région sert à apprendre, jamais à décider.** Le modèle a besoin de la région pour
isoler la pénalité du comité : sans elle, la pénalité se déplace dans le poids d'une
colonne d'apparence neutre (rapport, section 4). Une fois la pénalité retirée, la décision
ne lit que la cote R et les heures travaillées. Deux candidats de même dossier reçoivent la
même réponse, où qu'ils habitent. Nous avons écarté les seuils différents par groupe
(`ThresholdOptimizer`) : ils donnent un résultat voisin, mais la décision y dépend de la
région du candidat, ce qui est plus difficile à expliquer à un candidat refusé.

**Les heures travaillées comptent comme du mérite.** C'est ce que le comité valorisait une
fois ses biais retirés, et c'est ce que les retours du correcteur appuient. Leur poids
vient de l'historique, pas d'un réglage sur ces retours. C'est un choix de valeur : les
heures peuvent traduire l'effort, mais aussi la contrainte financière, et un candidat qui
ne peut pas travailler (santé, charge familiale) n'en tire aucun avantage. **L'institution
doit valider ce choix explicitement.**

**La préférence du comité pour les revenus élevés est retirée.** C'est un second biais,
indirect : les familles des régions éloignées ont des revenus plus bas. Pour une aide
financière, favoriser le revenu élevé ne se défend pas. Le modèle n'utilise pas non plus le
revenu dans l'autre sens (favoriser le besoin) : c'est une politique possible, mais elle
doit être décidée par l'institution, pas introduite par un modèle.

**Qui supporte le coût.** Le budget est fixe. Les bourses rendues aux régions éloignées
sont retirées à des candidats des grands centres, surtout ceux à revenu élevé et à peu
d'heures travaillées, que le comité favorisait (rapport, section 7). Ce transfert est la
correction elle-même : il doit être annoncé, pas découvert.

## 2. Rôles et responsabilités

| Rôle | Qui | Responsabilité |
|---|---|---|
| Propriétaire du modèle | Direction de l'aide financière | Répond des décisions ; approuve chaque version et chaque choix de la section 1. |
| Équipe de science des données | Équipe interne | Entraîne, documente, exécute la surveillance à chaque cycle, signale toute alerte. |
| Réviseur indépendant | Audit interne ou tiers, hors de la ligne hiérarchique de l'équipe | Refait l'audit avant chaque mise en production et une fois par an ; a accès au code et aux données. |
| Comité d'équité | Direction, conformité, représentants étudiants dont des régions éloignées | Reçoit les alertes rouges ; **peut suspendre le modèle** et revenir à l'examen humain. |
| Responsable de la protection des renseignements personnels | Désigné par l'institution | Veille aux obligations d'information et de recours de la section 5. |
| Réviseurs de dossiers | Personnel formé, sans accès à la région du candidat | Examine les dossiers proches du seuil et les demandes de révision. |

Règle d'arrêt : une alerte rouge sur la pénalité régionale, sur l'écart entre les groupes
ou sur le budget suspend l'envoi des décisions du cycle jusqu'à l'avis du comité d'équité.

## 3. Indicateurs de surveillance

Calculés par `monitoring.py` sur chaque cycle de décisions, avec un intervalle à 95 % par
rééchantillonnage. Une alerte ambre déclenche une analyse par l'équipe dans la semaine ;
une alerte rouge applique la règle d'arrêt et saisit le comité d'équité.

| Indicateur | Définition | Ambre | Rouge | Pourquoi |
|---|---|---|---|---|
| Taux d'octroi global | Part des candidats qui reçoivent la bourse | hors de 38 % à 42 % | hors de 36 % à 44 % | Le budget est une contrainte ferme. |
| Écart entre les groupes | Taux d'octroi des grands centres moins celui des régions éloignées | 0,06 | 0,10 | Le signal le plus direct ; nul pour le modèle corrigé. |
| Écart entre deux régions | Plus grande différence de taux entre deux régions | 0,12 | 0,18 | L'audit n'a vu que deux blocs ; un écart à l'intérieur d'un bloc doit rester détectable. |
| Pénalité régionale | Terme « région éloignée » d'un modèle logistique réajusté sur les décisions du cycle, en points de cote R | 0,35 | 0,70 | Mesure le biais lui-même, à dossier égal. Dans l'historique du comité, elle dépasse largement le seuil rouge. |
| Poids des heures appris sans le groupe | Poids d'une heure dans le même modèle ajusté sans la région | sous 0,115 | sous 0,09 | Une pénalité cachée fait baisser ce poids : il alerte même si la région n'est plus collectée. |
| Égalité des chances sur échantillon audité | Écart de taux de vrais positifs entre les groupes, contre l'étalon de la section 4 | 0,10 | 0,20 | Le seul indicateur mesuré contre un jugement indépendant du système. |
| Dérive des entrées | Indice de stabilité de la population (cote R, heures, revenu) contre la population de référence | 0,10 | 0,25 | Un changement de population peut invalider le poids appris. |

Deux précisions.

- **L'indicateur classique « les entrées prédisent-elles la région ? » n'est pas retenu.**
  Il est saturé dès le départ (la distance et le code postal donnent la région presque à
  coup sûr) : il ne peut pas bouger, donc pas alerter. La pénalité réajustée le remplace.
- **Sous-groupes.** Les taux par première génération, tranche de revenu, tranche d'heures
  et programme, croisés avec le groupe, sont publiés à chaque cycle (rapport, section 7).
  Ils n'ont pas de seuil automatique : les effectifs sont petits, et leur lecture revient
  au réviseur indépendant.

Démonstration : `python monitoring.py` rejoue le modèle sur des périodes simulées, puis
déploie par erreur une version à moitié corrigée. Les périodes normales restent au vert ;
la période fautive passe au rouge sur la pénalité et sur l'écart entre les groupes
(`figures/monitoring.png`).

## 4. Étalon de référence continu

En production, le système n'observe que ses propres décisions. Mesuré contre elles, un
modèle biaisé paraît équitable : c'est exactement ce qui est arrivé au modèle en production
(rapport, section 6).

À chaque cycle, un échantillon aléatoire de dossiers est donc réexaminé **à l'aveugle** par
un panel indépendant : sans la région, le code postal ni la distance, et sans connaître la
décision du modèle. Le panel dit, pour chaque dossier, s'il mérite la bourse. Ses réponses
servent à trois choses : calculer l'écart d'égalité des chances de la section 3, vérifier
que l'étalon « cote R et heures » tient toujours, et constituer peu à peu des données
d'entraînement qui ne viennent pas du comité historique. L'échantillon surreprésente les
dossiers proches du seuil, où se trouvent la plupart des erreurs probables
(`analysis/FINDINGS.md`, bloc 16).

## 5. Recours et transparence

**Ce qui est dit au candidat.** Avec la décision : qu'elle repose sur un traitement
automatisé, les éléments pris en compte (cote R, heures travaillées), et que ni sa région,
ni son code postal, ni son revenu n'entrent dans la décision. La règle tient en une ligne,
ce qui rend l'explication exacte et non approximative.

**Révision humaine.** Tout candidat refusé peut demander la révision de son dossier par un
réviseur qui a le pouvoir de modifier la décision. Les dossiers à moins d'un demi-point de
cote R du seuil sont examinés par un réviseur avant l'envoi de la décision, sans attendre
de demande.

**Correction des renseignements.** Le candidat peut faire corriger une cote R ou un nombre
d'heures erroné ; la décision est alors recalculée.

**Transparence publique.** Chaque année : la règle de décision, les taux d'octroi par
région, les indicateurs de la section 3 et le nombre de décisions modifiées en révision.

## 6. Cycle de vie

- **Versions.** Une version est une spécification (`RETAINED` dans `model_corrige.py`) et
  un commit. Les décisions d'un cycle sont archivées avec la version qui les a produites.
- **Avant toute mise en production.** Audit complet par le réviseur indépendant ;
  `python build.py` passe ; tous les indicateurs sont au vert sur le dernier cycle rejoué.
- **Réapprentissage.** Le poids des heures est réappris seulement sur des décisions de
  l'étalon continu, jamais sur les décisions du modèle lui-même (il n'y retrouverait que
  sa propre règle). Déclencheurs : dérive des entrées à l'ambre deux cycles de suite, ou
  désaccord croissant entre le panel et le modèle.
- **Réaudit.** Une fois par an, à tout changement de colonnes collectées, de population ou
  de budget, et après toute alerte rouge.
- **Retour en arrière.** Sur alerte rouge confirmée : retour à la dernière version au vert,
  ou à l'examen humain des dossiers si aucune ne l'est. Le modèle historique n'est jamais
  une solution de repli.

## 7. Risques résiduels

- **L'étalon du mérite reste une hypothèse.** Il s'accorde le mieux avec les retours du
  correcteur, obtenus sur un sous-ensemble ; il n'est pas prouvé. L'étalon continu de la
  section 4 est la réponse durable.
- **Les heures travaillées peuvent être déclarées à la hausse** une fois la règle connue.
  Elles doivent être vérifiables (relevés d'emploi) ou leur poids plafonné.
- **Les candidats qui ne peuvent pas travailler** ne sont pas vus par la règle. Voie
  prévue : la révision humaine, et une décision de l'institution sur un critère de besoin.
- **Données synthétiques.** Le comité y suit une règle simple et constante. Un comité réel
  serait moins régulier ; la pénalité pourrait varier selon les dossiers et la correction
  en forme fermée serait à revalider.
- **Deux blocs.** Un écart entre régions d'un même bloc ou sur un autre critère (langue,
  origine) n'est pas mesurable avec ces colonnes.
- **Les dossiers proches du seuil.** Aucune règle ne les départage de façon fiable ; c'est
  pourquoi ils vont à un réviseur.
