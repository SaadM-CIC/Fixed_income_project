# Fixed Income & Modélisation des taux d'intérêt en Python

Projet académique et pédagogique couvrant la valorisation obligataire, la construction d'une courbe des taux, les mesures de risque (duration, convexité, DV01), la valorisation d'un swap de taux (IRS) et le modèle de taux court de Vasicek.

Le code est volontairement simple : fonctions courtes en NumPy, notebooks Jupyter pour les explications, tests de cohérence financière avec `pytest`. Chaque étape suit le même plan : intuition, formules et hypothèses, implémentation, tests, interprétation.

> Projet réalisé dans le cadre de ma formation d'ingénieur à l'École Mohammadia d'Ingénieurs (EMI), filière Modélisation Mathématique et Informatique Scientifique.

---

## Contenu

| # | Notebook | Module | Sujet |
|---|---|---|---|
| 1 | `01_bond_pricing.ipynb` | `src/bonds.py` | Prix d'une obligation, YTM par dichotomie |
| 2 | `02_yield_curve.ipynb` | `src/curve.py` | Bootstrapping, taux zéro-coupon, forwards, interpolation, courbe US Treasury réelle |
| 3 | `03_risk_measures.ipynb` | `src/risk.py` | Duration, convexité, DV01, key-rate DV01, couverture |
| 4 | `04_interest_rate_swap.ipynb` | `src/swap.py` | IRS : jambes fixe et variable, taux swap au pair, DV01, couverture d'une obligation |
| 5 | `05_vasicek.ipynb` | `src/vasicek.py` | Vasicek : simulation, prix zéro-coupon, calibration, option sur zéro-coupon |

---

## Structure du dépôt

```
fixed-income-project/
├── README.md
├── requirements.txt
├── src/
│   ├── bonds.py        # cashflows, bond_price, ytm
│   ├── curve.py        # bootstrapping, interpolation, forwards, courbe US
│   ├── risk.py         # duration, convexité, DV01, key-rate DV01
│   ├── swap.py         # annuité, taux swap, valeur et DV01 d'un IRS
│   └── vasicek.py      # simulation, prix ZC, Monte Carlo, calibration
├── notebooks/          # un notebook par étape
├── tests/              # tests de cohérence (pytest)
├── data/
│   └── us_par_yields.csv   # snapshot des taux au pair US Treasury (FRED)
└── figures/            # graphiques exportés des notebooks
```

---

## Installation

```bash
python -m venv .venv
# Windows (PowerShell) : .venv\Scripts\Activate.ps1
# macOS / Linux        : source .venv/bin/activate
pip install -r requirements.txt
```

Bibliothèques utilisées : `numpy`, `pandas`, `matplotlib`, `jupyter`, `ipykernel`, `pytest`.

## Utilisation

Lancer les tests depuis la racine du projet :

```bash
pytest
```

Ouvrir les notebooks dans VS Code (extension Jupyter) ou avec `jupyter lab`, dans l'ordre `01` à `05`. Les notebooks importent les fonctions depuis le package `src/`; exécutez-les avec la racine du projet accessible au noyau Python.

Les données de marché sont des taux au pair du Trésor américain téléchargés depuis FRED (séries `DGS1`, `DGS2`, `DGS3`, `DGS5`, `DGS7`, `DGS10`) et sauvegardés dans `data/us_par_yields.csv`, ce qui rend les notebooks reproductibles sans connexion. Les tests, eux, reposent uniquement sur des données synthétiques déterministes.

---

## Résumé des étapes

### 1. Valorisation obligataire

Prix d'une obligation de flux $CF_k$ à des dates $t_k$, avec un taux unique $y$ et composition $m$ fois par an :

$$P(y)=\sum_k \frac{CF_k}{(1+y/m)^{m t_k}}$$

Le YTM est le taux $y$ qui égalise ce prix au prix de marché. Le prix est strictement décroissant en $y$, ce qui justifie une recherche par dichotomie.

### 2. Courbe des taux

- Facteurs d'actualisation $DF(t)=(1+z(t))^{-t}$ et taux zéro-coupon $z(t)$.
- Bootstrapping à partir de taux au pair $c_n$ : $DF(n)=\dfrac{1-c_n\sum_{k<n}DF(k)}{1+c_n}$.
- Taux forward : $f(t_1,t_2)=\left(\dfrac{DF(t_1)}{DF(t_2)}\right)^{1/(t_2-t_1)}-1$.
- Interpolation linéaire sur les taux zéro, application à la courbe US Treasury à plusieurs dates (courbe pentue, inversée) et suivi de la pente 10 ans − 2 ans.

### 3. Mesures de risque

- Duration de Macaulay $D=\sum t_k PV_k / P$, duration modifiée $D_{mod}=D/(1+y/m)$.
- Convexité $C=\frac{1}{P}\frac{\partial^2 P}{\partial y^2}$ et approximation de Taylor $\Delta P/P \approx -D_{mod}\Delta y+\tfrac12 C\,\Delta y^2$.
- DV01 analytique et par bump and reprice, DV01 sur courbe et key-rate DV01 par pilier.
- Exemple de couverture en DV01 et limite d'une couverture uniquement parallèle (scénarios de pentification et d'aplatissement).

### 4. Swap de taux (IRS)

Un swap payeur vaut « floater au pair moins obligation à taux fixe » :

$$A=\sum_i \tau\,DF(T_i),\qquad S=\frac{1-DF(T_n)}{A},\qquad V_{payeur}=N\cdot A\cdot(S-K)$$

La jambe variable se télescope en $N\,(1-DF(T_n))$. Le taux swap au pair coïncide, dans le cadre mono-courbe de ce projet, avec le taux au pair bootstrappé de l'étape 2. Le notebook compare aussi les profils de risque par pilier d'un swap et d'une obligation.

### 5. Modèle de Vasicek

$$dr_t=a(b-r_t)\,dt+\sigma\,dW_t$$

- Simulation par le schéma exact (loi gaussienne conditionnelle), comparée à Euler.
- Moyenne, variance et loi stationnaire $\mathcal N\!\left(b,\sigma^2/2a\right)$ vérifiées par simulation, probabilité de taux négatifs.
- Prix zéro-coupon en formule fermée $P(0,\tau)=A(\tau)e^{-B(\tau)r_0}$, avec $B(\tau)=\frac{1-e^{-a\tau}}{a}$, validé par Monte Carlo.
- Calibration sur une courbe de taux zéro (moindres carrés linéaires en $(r_0, R_\infty)$ et balayage sur $a$, $\sigma$ fixé) et estimation historique par régression AR(1) (mesure $\mathbb P$).
- Valorisation d'une obligation et d'un swap avec la courbe du modèle, et d'une option sur zéro-coupon par Monte Carlo.

---

## Tests

Les tests (`pytest`) vérifient notamment :

- une obligation au pair vaut son nominal, un zéro-coupon suit $N/(1+y)^T$, le YTM retrouve le taux de départ ;
- les obligations au pair sont revalorisées à 100 avec la courbe bootstrappée, une courbe plate reproduit l'étape 1, la cohérence des forwards ;
- duration d'un zéro-coupon égale à sa maturité, DV01 analytique contre bump, convexité contre dérivée seconde numérique, somme des key-rate DV01 contre DV01 parallèle ;
- taux swap au pair égal au taux au pair bootstrappé, valeur nulle au taux swap, télescopage de la jambe variable, swap égal à floater moins obligation ;
- moments de la simulation de Vasicek, limite $\sigma\to 0$, limite des taux longs, sensibilité $-B(\tau)$, formule fermée contre Monte Carlo, récupération des paramètres par calibration et par régression.

---

## Simplifications académiques et pratiques de marché

| Dans ce projet | Sur le marché |
|---|---|
| Taux unique $y$ puis courbe zéro-coupon sur taux au pair synthétiques ou du Trésor | Courbes construites sur dépôts, futures/FRA et swaps, avec des splines ou un modèle paramétrique |
| Une seule courbe pour actualiser et projeter | Multi-courbes : actualisation OIS (collatéral), projection sur l'indice (SOFR, €STR) |
| Composition simple, pas de conventions de jours | Act/360, 30/360, calendriers, coupons courus, prix clean et dirty |
| Duration par choc parallèle | Risque par buckets de maturité, duration effective pour les produits à options |
| Swap vanille sans risque de crédit | CSA, collatéral, XVA |
| Vasicek à un facteur, paramètres constants, taux négatifs possibles | Hull-White, G2++, LMM, SABR ; calibration sur options |
| Pas de prix de marché du risque | Distinction entre mesure historique $\mathbb P$ et mesure de valorisation $\mathbb Q$ |

## Limites

- La courbe US Treasury est un proxy : un swap USD se valorise sur une courbe SOFR/OIS.
- L'interpolation linéaire sur les taux zéro est une approximation, et les maturités manquantes (4, 6, 8, 9 ans) sont interpolées.
- Vasicek ne reproduit pas la courbe du jour (d'où Hull-White), ses courbes sont de forme limitée, et $\sigma$ est mal identifié sur une courbe seule.
- La couverture en DV01 est neutre pour un choc parallèle uniquement.

## Pistes d'extension

- Hull-White à un facteur (fit exact de la courbe initiale).
- CIR pour des taux positifs.
- Valorisation d'un cap/floor ou d'une swaption.
- Courbe OIS et courbe de projection séparées (cadre multi-courbes).
- Interpolation par splines ou Nelson-Siegel.

## Références

- O. Vasicek, *An Equilibrium Characterization of the Term Structure*, Journal of Financial Economics, 1977.
- J. Hull, *Options, Futures, and Other Derivatives*.
- D. Brigo et F. Mercurio, *Interest Rate Models: Theory and Practice*.

---

## Auteur

Saad, élève ingénieur en dernière année, École Mohammadia d'Ingénieurs (EMI), Modélisation Mathématique et Informatique Scientifique.

Projet à but pédagogique : les résultats ne constituent pas des conseils d'investissement.