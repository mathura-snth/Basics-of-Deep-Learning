
# PARTIE I du cours

## Motivation : ce que fait le deep learning aujourd'hui

Les premières diapos sont une démonstration d'applications, pas du contenu à calculer : détection et segmentation d'objets, comptage de foules, reconnaissance faciale (DeepFace), voitures autonomes, **diagnostic médical** (classification de cancers de la peau au niveau d'un dermatologue, *Nature* 2017 ; modèles de langage médicaux type Med-Gemini), assistants juridiques, traduction automatique (DeepL), ChatGPT (*Generative Pre-trained Transformer*), DeepDream, jeux de Go. Message : les réseaux de neurones profonds obtiennent des résultats spectaculaires dans la perception (vision, parole) et le langage.

Le cours situe aussi le **prix Turing 2018** (LeCun, Hinton, Bengio) et les **prix Nobel 2024** de physique et de chimie liés à l'IA.

## Place du deep learning dans l'IA

Emboîtement : **Deep Learning ⊂ Machine Learning ⊂ Intelligence artificielle**. Les grands domaines de l'IA vus dans le cours : algorithmique de l'IA (logique, recherche heuristique, planification) ; systèmes multi-agents ; robotique ; représentation des connaissances ; traitement automatique du langage ; interaction homme-machine ; vision et reconnaissance des formes ; **apprentissage automatique**.

Paradigmes d'apprentissage : **supervisé** (on connaît la cible d pour chaque x), **non supervisé** (pas de cible, on cherche la structure des données), **par renforcement** (récompenses), **par transfert / adaptation de domaine**. Les réseaux de neurones, les méthodes à noyaux, les méthodes bayésiennes en font partie.

## Qu'est-ce que le Machine Learning ?

Une tâche = apprendre une fonction qui transforme une **entrée** en **sortie** : classer un chiffre manuscrit, prédire le mot suivant, choisir le prochain coup au Go, segmenter une image, établir un diagnostic. Image géométrique du cours : le modèle réalise un « morphing » de l'espace d'entrée X vers l'espace de sortie Y, exprimé comme une **suite de transformations géométriques simples (les couches)**.

### Exemple fil rouge : régression linéaire (arbres)

Données : pour 13 arbres, hauteur (m) et diamètre (cm). On cherche un modèle pour **prévoir la hauteur à partir du diamètre** :

```
Hauteur = a · Diamètre + b        (a = pente, b = biais / ordonnée à l'origine)
```

On déplace la droite en faisant varier a et b jusqu'à ce qu'elle passe « au mieux » du nuage de points ; une fois a et b fixés, on peut **prévoir** la hauteur d'un arbre de diamètre jamais vu. Ceci résume tout le ML supervisé :
1) Choisir une famille de modèles paramétrés (ex: décider qu'on va utiliser un réseau de neurones à 2 couches cachées plutôt qu'une droite).
2) Mesurer l'erreur (calculer l'écart entre les prédictions du modèle et les vraies étiquettes grâce à une fonction de coût / de perte).
3) Ajuster les paramètres (par exemple via la descente de gradient et la rétropropagation) pour réduire cette erreur et trouver le meilleur modèle possible au sein de cette famille.   

### Limite : la grande dimension

Pour des problèmes réels de grande taille, la droite de régression **ne suffit pas**. Exemple : une image 400×400 = **160 000 nombres** en entrée (chat / autre ?) ; une image 1000×1000 = **10⁶ nombres**. Il faut des modèles **plus riches : les réseaux de neurones artificiels**.

### Extraction de caractéristiques et dimension cachée

On n'utilise pas les pixels bruts : on **extrait des caractéristiques** (*features*). Exemple du visage : 1000×1000 pixels = 1 000 000 de dimensions, mais un visage est décrit par 3 coordonnées cartésiennes + 3 angles d'Euler (pose) + environ 50 muscles (expression) + l'éclairage : la **dimension cachée est bien inférieure à \~56**. Une bonne représentation d'une image de visage contient donc beaucoup moins de 10⁶ valeurs. On passe de l'image brute (millions de pixels) à quelques dizaines de caractéristiques pertinentes (visage / autre, pose, expression, éclairage), puis on classe.

### Apprendre les caractéristiques : le deep learning

Plutôt que de concevoir les caractéristiques à la main, un **réseau profond** les apprend : extraction **hiérarchique** (des motifs simples vers des motifs abstraits), puis classification. Cela crée un **nouvel espace de représentation**. La question posée : *comment apprendre ces caractéristiques avec de tels réseaux ?* Réponse historique : les réseaux convolutifs entraînés par rétropropagation.

Petit point sur la rétropropagation : 
- c'est une méthode mathématique par laquelle le réseau « apprend de ses erreurs »
- le réseau est composé de milliers ou de millions de paramètres (les poids $w$) répartis sur plusieurs couches interconnectées. La question est : quel poids exact est responsable de cette erreur, et de combien doit-on le modifier ? Pour les couches cachées situées au milieu du réseau, on ne connaît pas la sortie idéale qu'elles auraient dû produire. C'est ce qu'on appelle le problème d'affectation du crédit
- deux phases :
    1) Forward pass : L'information (les données d'entrée) traverse le réseau de la première couche jusqu'à la dernière pour produire une prédiction
    2) Backward pass (ou rétropropagation) : On calcule d'abord l'erreur tout à la fin, sur la couche de sortie puis on remonte le réseau à l'envers (de la sortie vers l'entrée). Grâce à la règle de dérivation en chaîne (chain rule), l'erreur est propagée de couche en couche vers l'arrière. Chaque neurone calcule sa propre part de responsabilité dans l'erreur (le $\delta$ du neurone)
- Une fois que l'erreur a été rétrofusée à travers tout le réseau et que chaque neurone connaît son $\delta$, on utilise un algorithme d'optimisation (comme la descente de gradient) pour modifier tous les poids de connexions du réseau :
   $$\text{Nouveau poids} = \text{Ancien poids} - (\text{taux d'apprentissage}) \times (\text{responsabilité de l'erreur})$$

### Apprentissage dans les réseaux profonds : l'enjeu

Construire un espace de caractéristiques, c'est ce que font les noyaux dans les SVM ou les couches cachées d'un MLP, mais ici l'espace de représentation est construit par une architecture profonde. Difficultés : gradient qui se diffuse mal vers les premières couches, peu de données étiquetées, plus de minima locaux. D'où l'intérêt de l'apprentissage non supervisé couche par couche.

Petit point sur le SVM :
- le but d'un SVM est de tracer une ligne (en 2D), un plan (en 3D) ou un hyperplan (en dimension supérieure) pour séparer ces deux classes de manière optimale
- contrairement au Perceptron (qui s'arrête dès qu'il trouve une frontière qui sépare les données, quitte à ce qu'elle soit collée à un point), le SVM cherche la frontière qui maximise la marge : il s'installe le plus loin possible des points des deux classes
- les points qui se trouvent le plus près de la frontière et qui servent à « caler » cette séparation s'appellent les vecteurs de support
- dans un SVM, on utilise des fonctions mathématiques (les noyaux) pour projeter les données dans un espace où elles sont faciles à séparer.
- dans un MLP (réseau de neurones multicouches), ce sont les couches cachées qui réalisent cette transformation en tordant et en remodelant l'espace d'entrée pour construire elles-mêmes cet espace de caractéristiques abstrait.

---

## Le neurone formel

### Motivation biologique et ses limites

Les organismes vivants réalisent des tâches complexes (orientation, communication, comportement social). Leur puissance vient de la **connexion massive** (≈10¹⁴ connexions chez l'humain) d'un **grand nombre d'unités simples** (≈10¹¹ neurones). L'idée initiale (années 1940) était la neuromimétique, l'idée s'est avérée très féconde en mathématiques et en ingénierie : le neurone artificiel est **un modèle, pas une copie**, du neurone biologique.

### Définition

Un **neurone formel** est une unité de calcul qui reçoit des données d'entrée sous forme de vecteur et produit une **sortie réelle**, fonction des entrées et des poids des connexions. C'est un processeur élémentaire caractérisé par : signaux d'entrée ; poids des connexions ; fonction d'activation ; état d'activation interne ; fonction de transition ; état de sortie.


- Activation :  $a = w_0 + \sum_{i=1}^{D} w_i x_i = w^T x$   (avec $x_0 = 1$ : unité de biais)
- Sortie     :  y = f(a)


Le poids w₀ du **biais** permet de décaler la frontière de décision (sans biais, l'hyperplan passe obligatoirement par l'origine — voir les figures « sans biais / avec biais » de l'Adaline).

### Fonctions d'activation

Seuil (échelon), linéaire, sigmoïde, tanh, ReLU (voir Partie II, section 5). **Neurone « distance »** : au lieu d'un produit scalaire, le neurone calcule une distance entre l'entrée et son vecteur de poids (famille des fonctions à base radiale, RBF)

### Réseau de neurones

**Définition** : un réseau de neurones est un **graphe orienté valué** constitué d'unités réalisant des calculs élémentaires, structurées en couches successives, qui échangent de l'information par des connexions. Caractéristiques : architecture **parallèle** ; coopération de nombreuses unités simples ; un réseau est caractérisé par son **architecture** (graphe de connexions) et les **fonctions de ses éléments**.

---

## L'Adaline

**Adaline = Adaptive Linear Element** (Stanford, Bernard Widrow). Un neurone **linéaire** adaptatif ; l'unité x₀ = 1 est l'**unité de biais**.

### Modèle pour la discrimination à 2 classes
L'objectif est de séparer des données en deux catégories distinctes, notées $\mathcal{C}_1$ et $\mathcal{C}_2$.
Pour ce faire, le modèle calcule une valeur globale via la formule : 
$$\psi(x, w) = w_0 + w^T x$$
- $\psi(x, w) > 0 \quad \text{si} \quad x \in \mathcal{C}_1$
- $\psi(x, w) < 0 \quad \text{si} \quad x \in \mathcal{C}_2$
- $\psi(x, w) = 0 \quad \text{: frontière de décision (hyperplan)}$


Codage des cibles : d = +1 pour C₁, d = −1 pour C₂. La **sortie calculée y = wᵀx est linéaire** (pas de seuil pendant l'apprentissage) ; la décision se fait sur le signe.

### Fonction d'erreur

Si z(n) = (x(n), d(n)) est l'exemple considéré à l'itération n :


- **Erreur quadratique instantanée** :  $R̃(n)(w) = ( d(n) − wᵀ x(n) )²$ (C'est l'erreur commise par le modèle pour un seul exemple $n$)
- **Erreur quadratique moyenne (MSE)** : $R̃(w) = (1/N) Σ_{n=1}^{N} R̃(n)(w)$ (C'est la moyenne de toutes les erreurs instantanées commises sur l'ensemble des $N$ exemples du jeu de données)


### Règle d'apprentissage : descente de gradient stochastique

Principe (plus grande pente) : à l'instant t, les poids valent w(t) ; on présente l'exemple (x(n), d(n)) et on corrige :

$w(t+1) = w(t) − ε(t) · ∇_w R̃(n)(w)$

Dérivation :  $∇_w R̃(n) = ∂/∂w ( d − wᵀx )² = 2 ( d − wᵀx ) · (−x) = −2 ( d − wᵀx ) x$

Donc :  $w(t+1) = w(t) + 2ε(t) ( d(n) − y(n) ) x(n)$      (le 2 est absorbé dans ε)

Interprétation : on corrige les poids **proportionnellement à l'erreur (d − y)** et à l'entrée x. Si l'erreur est nulle, rien ne bouge. Cette règle s'appelle **règle du gradient stochastique**, **règle de Widrow-Hoff**, **règle delta**, ou **µ-LMS (Least Mean Square)**.

### Algorithme de l'Adaline

1. Initialiser aléatoirement w.
2. Choisir au hasard un couple (x(n), d(n)).
3. Déterminer l'erreur d − y.
4. Calculer l'approximation du gradient (gradient instantané).
5. Adapter les poids : w ← w − ε(t) ∇R̃(n).
6. Répéter de 2 à 5 jusqu'à obtenir une erreur acceptable.

### Exemple : apprendre la fonction OU

Table de vérité en codage ±1 : (1,1)→+1 ; (1,−1)→+1 ; (−1,1)→+1 ; (−1,−1)→−1. On ajoute l'entrée de biais x₀ = 1 et on cherche w₀, w₁, w₂ tels que la sortie a = w₀ + w₁x₁ + w₂x₂ ait le bon signe. Après apprentissage, la droite **w₁x₁ + w₂x₂ + w₀ = 0** sépare l'unique point négatif (−1,−1) des trois points positifs dans le plan (x₁, x₂). Une telle droite existe : le problème est **linéairement séparable**.

### Limites : séparabilité linéaire

**Définition** : deux classes d'objets décrites dans un espace de dimension n sont **linéairement séparables** si elles se trouvent de part et d'autre d'un hyperplan de l'espace de représentation. Un Adaline (ou un perceptron) ne peut résoudre que des problèmes linéairement séparables : par exemple le **XOR** est impossible.

**Contournement 1 — prétraitement polynomial** : on ajoute des entrées non linéaires (x, x², produits…) devant l'Adaline ; dans le nouvel espace, une frontière devient linéaire alors qu'elle est une **ellipse** dans l'espace d'origine. Mais il faut choisir les prétraitements à la main.

**Contournement 2 — Madaline (Multi-Adaptive Linear Element)** : un ensemble d'Adalines **en parallèle** (une par classe/sortie, p classes → p Adalines) ; combinaison des sorties pour obtenir des frontières plus complexes. Étape intermédiaire vers le perceptron multicouche.

---

## Le Perceptron

Le perceptron n'est pas un modèle unique mais une **famille d'algorithmes** : une machine adaptative pour des problèmes de **classification (discrimination)**.

### Architecture

- **La rétine R** : reçoit l'information du monde extérieur.
- **Les cellules d'association φ\_i** : chaque cellule a une fonction de transition φ\_i : R → ℝ définie sur la rétine (prétraitements fixes, non appris).
- **La cellule de décision** : $ψ(x,w) = f( w₀ + Σ w\_i * φ\_i(x) ) = f(wᵀφ)$, avec $φ = (1, φ₁(x), …, φ\_D(x))$ et f un **échelon** : f = +1 si l'argument ≥ 0, −1 sinon.

Différence essentielle avec l'Adaline : la sortie passe par le **seuil f** (sortie binaire), et l'erreur est définie à partir de ce seuil.

### Cas 2 classes

```
C₁ = { x : ψ(x,w) = +1 }     C₂ = { x : ψ(x,w) = −1 }
```

Cibles d(n) = +1 si x(n) ∈ C₁ ; d(n) = −1 si x(n) ∈ C₂. Un exemple est **bien classé** ssi wᵀφ(x(n)) · d(n) > 0 (même signe que la cible).

**Erreur** : on ne pénalise que les exemples mal classés :

- $R̃(n)(w) = − d(n) · wᵀφ(x(n))$     si x(n) est mal classé (le produit est alors < 0, donc $R̃ > 0$)
- $R̃(w)    = − (1/N) Σ_{n mal classés} d(n) · wᵀφ(x(n))$     (critère du perceptron)


**Gradient et mise à jour** :

- $∇_w R̃(n) = − d(n) · φ(x(n))$
- $w(t+1) = w(t) − ε(t) ∇_w R̃(n) = w(t) + ε(t) · d(n) · φ(x(n))$     (si mal classé)


Critère d'arrêt : taux de classification satisfaisant.

### Algorithme (2 classes)

1. Initialiser aléatoirement w(0).
2. Choisir au hasard un couple (x(n), d(n)).
3. Calculer la sortie du perceptron f(wᵀφ) et la comparer à d(n).
4. Si x(n) est **bien classé** (f(wᵀφ) = d(n)) : w(t+1) = w(t). Si **mal classé** : w(t+1) = w(t) + ε(t) · φ(x(n)) · d(n).
5. Répéter de 2 à 4 jusqu'à obtenir une erreur acceptable.

Intuition : si un point de C₁ (d = +1) est mal classé, on **ajoute** une fraction de φ à w, ce qui tourne l'hyperplan pour que ce point passe du bon côté ; si un point de C₂ est mal classé, on **retranche**.

### Cas p classes (C₁ … C\_p)

Une unité de sortie par classe ; vecteur de poids w\_i pour la classe i. La classe prédite est celle de **plus grande activation** :

```
ψ(x,w) = C_k  ⟺  k = argmax_i  w_iᵀ φ(x)
```

Si x(n) ∈ C\_i mais le réseau répond C\_j (j ≠ i) :

```
w_i(t+1) = w_i(t) + ε(t) φ(x(n))     (on renforce la bonne classe)
w_j(t+1) = w_j(t) − ε(t) φ(x(n))     (on pénalise la classe gagnante à tort)
w_l(t+1) = w_l(t)  pour tout l ≠ i, j  (inchangé)
```

Si bien classé : aucun changement.

### Théorème de convergence du perceptron

« **Si un ensemble d'exemples est linéairement séparable, alors l'algorithme d'apprentissage du perceptron converge vers une solution correcte en un nombre fini d'itérations.** ». Attention : le théorème ne dit rien si les données ne sont **pas** séparables (l'algorithme oscille indéfiniment), et ne garantit pas la meilleure séparation, seulement **une** séparation.

### Perceptron vs Adaline : synthèse

|  | Perceptron | Adaline |
| --- | --- | --- |
| Erreur calculée sur | sortie **seuillée** (±1) | sortie **linéaire** (avant seuil) |
| Critère | exemples mal classés | erreur quadratique sur **tous** les exemples |
| Mise à jour | seulement si erreur de classification : w += ε d φ | toujours : w += 2ε (d − y) x |
| Solution | minimise le **nombre d'erreurs** (séparation quelconque valide) | séparation la plus **robuste** (moindres carrés) |
| Convergence | finie si séparable | descente de gradient sur un critère quadratique |

### Du perceptron au perceptron multicouche

Tous ces modèles sont limités aux problèmes linéairement séparables. La critique de Minsky & Papert (1969) a freiné la recherche ; elle ne concerne que le monocouche. Le **problème d'affectation du crédit** (on ne connaît pas la cible des unités cachées) est résolu par la **rétropropagation du gradient** : c'est le sujet de la Partie II.

---

## Checklist de révision (Partie I)

- Définir le neurone formel (McCulloch & Pitts) et le rôle du **biais**.
- Expliquer pourquoi on extrait des **caractéristiques** et ce qu'est la **dimension cachée** (exemple du visage).
- Citer les 3 raisons du retour du deep learning : GPU, Big Data, algorithmes.
- Écrire et **dériver** la règle de Widrow-Hoff (LMS) : ∇R̃ = −2(d−wᵀx)x.
- Écrire la règle du perceptron et expliquer pourquoi on n'ajuste que sur les exemples mal classés.
- Énoncer le **théorème de convergence** du perceptron et ses conditions.
- Comparer Perceptron et Adaline (tableau ci-dessus).
- Définir la **séparabilité linéaire** ; donner l'exemple du XOR ; citer les deux remèdes (prétraitement polynomial, Madaline) et la vraie solution (MLP + rétropropagation).

---

# PARTIE II du cours

## Rappels : Perceptron vs Adaline

Problème à 2 classes (A et B), séparées par un hyperplan.

- **Perceptron** : la sortie calculée passe par un seuil (échelon), et l'erreur est calculée *après* le seuil. La solution trouvée **minimise le nombre d'erreurs** de classification, mais s'arrête dès que tout est bien classé : la droite peut être collée à une classe.
- **Adaline** : l'erreur est calculée sur la sortie linéaire *avant* le seuil (règle des moindres carrés, LMS). Elle cherche la séparation la **plus robuste** (marge « moyenne » entre les classes), au prix de ne pas minimiser strictement le nombre d'erreurs.

À retenir : Perceptron = erreur de classification ; Adaline = erreur quadratique continue (dérivable, donc minimisable par gradient).

---

## Pourquoi les réseaux multicouches ?

### Le problème de l'affectation du crédit (*credit assignment problem*)

On a un réseau en couches et un ensemble d'exemples entrée-sortie :

$D_N = {(x(1),d(1)), ..., (x(N),d(N))},   x(n) ∈ ℝ^D,  d(n) ∈ ℝ^K$

- Pour la **couche de sortie** (poids W2), on connaît la cible : on peut appliquer la règle du perceptron (méthode de mise à jour des poids basée sur l'erreur de classification).
- Pour les **couches cachées** (poids W1), on **ne connaît pas la sortie désirée** des unités cachées : on ne peut donc pas leur appliquer l'algorithme du perceptron. Comment répartir « la faute » de l'erreur de sortie entre les poids de la couche cachée ? C'est le crédit à affecter.
- **Solution : la rétropropagation du gradient** (*Gradient Back-Propagation*, GBP).

---

## Le Perceptron Multi-Couches (MLP)

### Théorème de Kolmogorov (1957)

Toute fonction continue `f(x)` définie sur \[0,1\]ⁿ (à $n$ variables d'entrée) peut s'exprimée sous forme d'une combinaison de fonctions plus simples qui ne dépendent que d'une seule variable :

$f(x) = Σ_{i=1}^{2n+1}  F_i( Σ_{j=1}^{n}  G_ij(x_j) )$

où les `G_ij` et `F_i` sont des fonctions continues d'**une seule variable**. Interprétation réseau : une couche *cachée* de 2n+1 groupes d'unités qui calculent les G, puis des sommes, puis les F. Un réseau de neurones peut approximer les `G_ij` et `F_i` avec des fonctions de la forme `f(w₀ + wᵀx)`.

Exemple de la diapo : `f(x) = ¼[(x₁+x₂)² − (x₁−x₂)²]`. En développant : (x₁+x₂)² − (x₁−x₂)² = 4x₁x₂, donc f(x) = **x₁·x₂**. Le produit de deux variables se décompose en fonctions d'une seule variable (des carrés) et de sommes.

### Architecture

- Comme le perceptron/Madaline, **plus des couches intermédiaires (cachées)**.
- Couches externes : **entrée** (I unités), **sortie** (O unités). Couches internes : **cachées** (H unités).
- Notation `<I | H | O>` ; exemple `<6 | 4 | 2>` = 6 entrées, 4 cachées, 2 sorties.
- Les unités d'entrée sont une simple distribution des données ; il y a un **biais** (unité constante égale à 1) sur chaque couche.

### Notations Amont / Aval

- `Amont(k)` : ensemble des unités dont la **sortie sert d'entrée** à l'unité k.
- `Aval(k)` : ensemble des unités qui **utilisent la sortie** de k comme entrée.
- Exemple diapo 7 (unités 0–5 entrées, 6–9 cachées, 10–11 sorties) : Amont(7) = {0,1,2,3,4,5}, Aval(7) = {10,11}, Amont(2) = ∅ (entrée), Aval(11) = ∅ (sortie).

### Calcul avant (propagation)

Unité cachée j (poids de la couche 1, `w_ji^(1)`) :

- Activation :  $a_j = w_j0^(1) + Σ_{i ∈ Amont(j)} w_ji^(1) · x_i$
- Sortie     :  $z_j = f(a_j)$

Unité de sortie k (poids de la couche 2) :

$a_k = w_k0^(2) + Σ_{j ∈ Amont(k)} w_kj^(2) · z_j$
$y_k = f(a_k)$

En combinant :

$y_k = f( w_k0^(2) + Σ_{j=1}^{M} w_kj^(2) · f( w_j0^(1) + Σ_{i=1}^{D} w_ji^(1) · x_i ) )$

(D entrées, M unités cachées, K sorties ; `w_j0` = poids du biais.) C'est une composition de fonctions : un MLP est une fonction `y(x, w)` paramétrée par les poids.

Conclusion :
- Mathématiquement, un réseau de neurones multicouches n'est rien d'autre qu'une composition de fonctions (on applique une fonction, puis on réapplique une autre fonction à l'intérieur de la première, et ainsi de suite).
- Le réseau tout entier forme une grande fonction globale notée $y(x, w)$, dont le comportement dépend entièrement des poids ($w$).
- L'apprentissage consistera justement à trouver les valeurs idéales de ces poids pour que cette fonction géante transforme correctement l'entrée $x$ en la bonne sortie $y$. 

---

## Apprendre à partir d'exemples

### Risque théorique (erreur de généralisation)

$R(w) = ∫ L( y(x,w), d ) dp(x,d)$

- `L` = fonction de perte (loss) mesurant l'écart entre sortie et cible.
- `p(x,d) = p(x)·p(d|x)` : distribution **fixe mais inconnue**.
- Apprendre = trouver `w* = argmin_w R(w)`.
- Problème : `p(x,d)` est inconnue, donc **R(w) n'est pas calculable**. On dispose seulement d'un échantillon `D_N` de N exemples i.i.d. tirés de p(x,d).

### Minimisation du risque empirique (ERM)

Principe d'induction : on remplace l'intégrale par une moyenne sur les exemples :

$R̃(w) = (1/N) Σ_{n=1}^{N} L( y(x(n), w), d(n) ) = (1/N) Σ_n R(n)(w)$

`R(n)(w)` est le **risque empirique local** (l'erreur sur le n-ième exemple). L'apprentissage devient `w* = argmin R̃(w)`.

### Descente de gradient

On met à jour les poids dans la direction opposée au gradient :

$w_ji(t+1) = w_ji(t) − ε(t) · ∂R(w)/∂w_ji$

$Δw_ji(t) = − ε(t) · ∂R(w)/∂w_ji$

*N.B. : $∂R(w)/∂w_ji$ c'est le gradient, il indique la direction de la plus grande pente ascendante, c'est-à-dire là où l'erreur augmente le plus vite, or nous on veut l'inverse : réduire l'erreur au max*

- `ε` > 0 : pas d'apprentissage (learning rate).
- Intuition : si la pente en w₀ est **positive**, on va à **gauche** ; si elle est **négative**, on va à **droite** ; si elle est nulle, on est sur un extremum. Image du skieur : on suit la plus grande pente descendante.
- Si ε est trop grand, on oscille/diverge ; trop petit, la convergence est très lente. Sur une surface non convexe, on peut tomber dans un **minimum local**.

### Fonction de coût : erreur quadratique moyenne (MSE)

$R(w) = (1/N) Σ_n R(n)(w),     R(n)(w) = ½ Σ_{k=1}^{K} ( y_k(n) − d_k(n) )²$

Le facteur ½ sert uniquement à simplifier la dérivée (le 2 de la puissance s'annule).

---

## Fonctions de transfert / d'activation

| Fonction | Formule | Dérivée | Remarques |
| --- | --- | --- | --- |
| **Sigmoïde (logistique)** | f(x) = 1/(1+e^(−x)) | f'(x) = f(x)(1−f(x)) | sortie ∈ \]0,1\[, non centrée en 0, sature |
| **Tanh** | f(x) = (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ) | 1 − f(x)² | sortie ∈ \]−1,1\[, centrée en 0, sature |
| **ReLU** | f(x) = max(0, x) | 1 si x>0, 0 si x<0 | non saturante côté positif |

La fonction d'activation doit être **non linéaire** : sans cela, empiler des couches revient à une seule transformation linéaire. Le choix de dérivées simples (sigmoïde : f' = f(1−f)) rend la rétropropagation peu coûteuse.

### ReLU (Rectified Linear Unit) -> f(x) = max(0,x)

**Avantages :**

- **Activation parcimonieuse** : dans un réseau initialisé aléatoirement, environ 50 % des unités cachées ont une sortie non nulle.
- **Meilleure propagation du gradient** : moins de problèmes de gradient évanescent que les sigmoïdes, qui saturent des deux côtés (là où f' ≈ 0).
- **Calcul efficace** : comparaison, addition, multiplication seulement.
- **Invariance d'échelle** : max(0, a·x) = a·max(0, x) pour a>0.

**Problèmes :**

- Non dérivable en 0 (on choisit arbitrairement la dérivée 0 ou 1).
- Non centrée en zéro, non bornée.
- **Dying ReLU** : un neurone poussé à être inactif pour toutes les entrées (a<0) a un gradient nul ; aucun gradient ne le traverse, il n'apprend plus et « meurt ». Atténuation : les **leaky ReLU**.

**Variantes :**

- *Linéaires par morceaux* : Leaky ReLU (`f(x)=x` si x>0, sinon `0,01·x`) ; Parametric ReLU (`f(x)=x` si x>0, sinon `a·x`, avec **a appris**).
- *Non linéaires* : **GELU** `f(x)=x·Φ(x)` (Φ = fonction de répartition de la loi normale centrée réduite ; approx. `0,5x(1+tanh[√(2/π)(x+0,044715x³)])`) ; **SiLU** `f(x)=x·sigmoïde(x)` ; **SmoothReLU** `f(x)=ln(1+e^{kx})/k` ; **ELU** (`x` si x>0, sinon `a(eˣ−1)`).

---

## Frontières de décision

Le pouvoir de représentation dépend du nombre de couches :

- **Sans couche cachée** (1 couche) : un **hyperplan** (demi-espace).
- **1 couche cachée** : un **ET** de plusieurs hyperplans → **région convexe**.
- **2 couches cachées** : un **OU** de régions convexes → **région de forme arbitraire** (non convexe, avec trous).

Les unités cachées de la 1ʳᵉ couche découpent l'espace en hyperplans ; la 2ᵉ les combine (ET) ; la couche de sortie combine ces régions (OU).

---

## Fonctions d'erreur

- MSE : $R_mse(w) = ½ Σ_n ( d(n) − y(n) )²$
- MSE pondérée : $R(w) = Σ_n (d(n) − y(n))ᵀ Σ⁻¹ (d(n) − y(n))$,   avec $Σ = (1/N) Σ_n (d(n) − y(n))(d(n) − y(n))ᵀ$
- Multiple logistique (entropie croisée binaire) : $R(w) = Σ_n Σ_i [ d_i log(d_i/y_i) + (1−d_i) log((1−d_i)/(1−y_i)) ]$
- Log-vraisemblance (softmax) : $R(w) = Σ_n d(n) log( d(n)/p(n) ),   p(n) = e^{y(n)} / Σ_j e^{y(j)}$


- **MSE** : régression, ou classification simple.
- **MSE pondérée** : tient compte de la covariance des erreurs (distance de Mahalanobis).
- **Entropie croisée / log-vraisemblance** : adaptée à des sorties interprétées comme des probabilités ; **softmax** transforme les sorties en probabilités positives de somme 1.

---

## La rétropropagation du gradient

### Notations

- $s_i$ : C'est la sortie d'un neurone $i$. Si c'est un neurone d'entrée, c'est simplement la donnée brute $x_i$.
- $a_j$ : L'activation du neurone $j$, c'est-à-dire la somme pondérée de toutes ses entrées en amont ($a_j = \sum w_{ji} s_i$).
- $s_j = f(a_j)$ : La sortie finale du neurone $j$ après application de la fonction d'activation non linéaire $f$.
- $R(n)$ : Le coût (l'erreur) commis sur un exemple $n$, mesuré ici par l'erreur quadratique $\frac{1}{2} \sum_k (y_k - d_k)^2$.  

### Dérivation pas à pas
Pour savoir de combien l'erreur $R$ change lorsqu'on modifie un poids précis $w_{ji}$.

**Étape 1 — règle de la chaîne.** Le poids `w_ji` n'agit sur R que via l'activation `a_j` :

$∂R/∂w_ji = (∂R/∂a_j) · (∂a_j/∂w_ji) = δ_j · s_i$

avec $δ_j = ∂R/∂a_j$ (le « delta » de l'unité j) et $∂a_j/∂w_ji = s_i$.

Donc on a : $$\frac{\partial R}{\partial w_{ji}} = \delta_j \cdot s_i$$

**Étape 2 — delta d'une unité de sortie (j ∈ Output).** Comme `y_j = f(a_j)` :
Pour un neurone qui se trouve tout à la fin du réseau (la sortie), on connaît directement la cible $d$.

En appliquant la règle de la chaîne entre l'erreur $R$, la sortie du neurone $y_j$ et son activation $a_j$, on obtient :

$δ_j = (∂R/∂y_j)(∂y_j/∂a_j) = (y_j − d_j) · f'(a_j)$

**Étape 3 — delta d'une unité cachée (j ∉ Output).**
Pour un neurone situé au milieu du réseau (couche cachée), on ne connaît pas sa cible directe.

L'activation `a_j` influence R uniquement à travers les unités h de l'aval :


$δ_j = ∂R/∂a_j = Σ_{h∈Aval(j)} (∂R/∂a_h)(∂a_h/∂a_j) = Σ_h δ_h · w_hj · f'(a_j)$

$δ_j = f'(a_j) · Σ_{h∈Aval(j)} w_hj · δ_h$


(car `a_h = Σ w_hj f(a_j)+…`, donc `∂a_h/∂a_j = w_hj f'(a_j)`.)

Interprétation : L'erreur d'un neurone caché est calculée en récupérant les erreurs ($\delta_h$) des neurones de la couche suivante, en les pondérant par les poids des connexions qui les relient ($w_{hj}$), puis en multipliant le tout par la dérivée $f'(a_j)$. C'est ainsi que l'erreur « remonte » du futur vers le passé.

**Étape 4 — mise à jour :**

Une fois que l'on connaît le delta ($\delta_j$) de chaque neurone grâce aux étapes précédentes, on met à jour le poids $w_{ji}$ en allant dans la direction opposée au gradient :

$w_ji(t+1) = w_ji(t) − ε(t) · δ_j · s_i$

### Interprétation

- **Propagation avant** : on calcule les activations et les sorties de la couche d'entrée vers la sortie.
- **Rétropropagation** : les deltas sont calculés de la **sortie vers l'entrée** ; l'erreur d'une unité cachée est la somme des erreurs des unités aval, **pondérée par les poids** qui les relient, puis multipliée par `f'(a_j)`. C'est ainsi que le crédit est réparti.
- Si `f'(a_j) ≈ 0` (saturation), le delta est étouffé : c'est la source du **gradient évanescent**.

### Algorithme d'apprentissage du MLP

1. Initialiser aléatoirement les poids W₀.
2. Choisir au hasard un couple (x(n), d(n)).
3. Calculer l'état du réseau par propagation : `a_j = w_j0 + Σ_i w_ji x_i`, `x_j = f(a_j)`.
4. Calculer le gradient : δ\_j (sortie) = f'(a\_j)(y\_j − d\_j) ; δ\_j (cachée) = f'(a\_j) Σ\_h w\_hj δ\_h.
5. Adapter les poids : `w_ji ← w_ji − ε(t) δ_j s_i`.
6. Répéter de 2 à 5 jusqu'à obtenir une erreur acceptable.

C'est une descente de gradient **stochastique** (un exemple tiré au hasard à la fois). Variante *batch* : on somme les gradients sur tous les exemples avant de mettre à jour.

**Complexité** (MLP à 2 couches) : nombre de paramètres `W = (D+1)·H + (H+1)·O` (D entrées, H cachées, O sorties, +1 pour les biais) ; le coût d'une passe avant/arrière est en O(W).

---

## Exemple : le XOR

Données (codage −1/+1) : (1,1)→−1 ; (1,−1)→+1 ; (−1,1)→+1 ; (−1,−1)→−1. Un perceptron simple ne peut pas séparer ces classes (non linéairement séparables).

**Idée du réseau de la diapo :** deux unités cachées définissent **deux droites parallèles** (ex. x₁+x₂+0,5 = 0 et x₁+x₂+1,5 = 0 sur la diapo). La bande entre les deux droites contient une classe, l'extérieur l'autre ; la sortie combine les deux unités cachées (poids 1,0 et −1,0, biais 0,5).

Construction équivalente que vous pouvez vérifier à la main : h₁ = signe(x₁+x₂+1), h₂ = signe(x₁+x₂−1), sortie = signe(h₁ − h₂ − 1).

- (1,1) : h₁=+1, h₂=+1 → 0−1 <0 → −1 ✓
- (1,−1) et (−1,1) : h₁=+1, h₂=−1 → 2−1 >0 → +1 ✓
- (−1,−1) : h₁=−1, h₂=−1 → 0−1 <0 → −1 ✓

**Leçon :** une couche cachée transforme l'espace d'entrée en un espace où le problème devient linéairement séparable.

**Exercice diapo 36** (à refaire seul) : réseau 2 entrées – 2 cachées – 2 sorties avec ReLU et les poids affichés. Procédure : (1) passe avant pour un exemple (calculer a, ReLU, puis y) ; (2) δ de sortie = (y−d)·f'(a) avec f'=1 si a>0 sinon 0 ; (3) δ cachés = f'(a\_j)·Σ w\_hj δ\_h ; (4) Δw = −ε δ\_j s\_i.

---

## 10. Organisation des données

### Codage des sorties (classification à C classes)

| Codage | Bleu | Blanc | Rouge |
| --- | --- | --- | --- |
| **1-parmi-C** (0/1) | 1 0 0 | 0 1 0 | 0 0 1 |
| **1-parmi-C** (−1/1) | 1 −1 −1 | −1 1 −1 | −1 −1 1 |
| **1-parmi-(C−1)** (0/1) | 1 0 | 0 1 | 0 0 |
| **Thermomètre** (0/1) | 1 1 1 | 0 1 1 | 0 0 1 |

- 1-parmi-C : une sortie par classe, la classe prédite est la sortie maximale. Par exemple, pour représenter le Rouge avec le codage $(0/1)$, seul le troisième neurone s'allume : 0 0 1. Pour désigner la classe finale prédite par le modèle, on regarde quel neurone a la sortie maximale
- 1-parmi-(C−1) : une sortie de moins (la dernière classe = tout à 0). Si tous les neurones de sortie renvoient $0$, le modèle comprend implicitement qu'il s'agit de la dernière classe (celle qui n'a pas de neurone dédié).
- **Thermomètre** : à utiliser quand les classes sont **ordonnées** (classement de classes). Ce type de codage est utilisé uniquement lorsque les classes possèdent un ordre logique ou une hiérarchie.

### Normalisation des entrées

Sans normalisation, les variables à grande échelle (ex. altitude en milliers vs pente en dizaines) dominent. Résultat illustré diapo 32 : sans normalisation la précision stagne, avec elle l'apprentissage converge.

Centrer-réduire :

$x(n) ← ( x(n) − x̄ ) / σ,    x̄ = (1/N) Σ_n x(n),    σ² = (1/N) Σ_n ( x(n) − x̄ )²$

Pourquoi :
1) les entrées centrées-réduites évitent de **saturer** les sigmoïdes (zones où f' ≈ 0) ;
2) la surface d'erreur est plus « ronde » : quand les paramètres ont des échelles différentes, le gradient du plus grand domine la mise à jour (ellipses allongées, trajectoire en zigzag) ; avec des échelles égales, tous les paramètres sont mis à jour dans des proportions comparables.

### Initialisation des poids

Objectif : que l'activation `a_i = Σ_{j∈Amont(i)} w_ij x_j` ait une variance ≈ 1 pour ne pas saturer la sigmoïde dès le départ (car activation ( = x dans la formule de la sigmoïde) importante (négative ou postive) implique sigmoide aplatie car tend vers 0 ou 1)

Hypothèses : entrées centrées-réduites (E\[x\]=0, Var\[x\]=1), poids indépendants des entrées et de moyenne nulle (E\[w\]=0).

$E[a_i]   = Σ_j E[w_ij] E[x_j] = 0$

$Var[a_i] = Σ_j E[w_ij²] E[x_j²] = |Amont(i)| · σ²[w_ij]$

On veut `Var[a_i] = 1`, donc :

$σ²[w_ij] = 1 / |Amont(i)|     ⟹     σ[w_ij] ∝ |Amont(i)|^(−1/2)$

On tire donc les poids dans une loi uniforme :

$w_ij ∈ [ −k/√|Amont(i)| , +k/√|Amont(i)| ],   0,5 < k < 2$

Avec la fonction recommandée `f(x) = 1,71·tanh(2x/3)`. Moins il y a d'entrées, plus les poids peuvent être grands, et inversement. *Ne jamais initialiser tous les poids à 0 ou à la même valeur* : toutes les unités cachées resteraient identiques (symétrie).

---

## Approximation universelle et profondeur

- **Théorème de Hornik (1989)** : un réseau à trois couches (une cachée) avec sorties linéaires et un nombre suffisant d'unités cachées saturantes croissantes peut approcher arbitrairement toute fonction mesurable bornée d'un espace de dimension finie vers un autre. **Un RNA est un approximateur universel.**
- **Limites** : théorème **purement existentiel** — il ne dit ni combien de neurones cacher, ni comment trouver les poids, ni que l'apprentissage convergera. En pratique, plusieurs couches et sorties non linéaires peuvent être plus commodes.
- **Intérêt de la profondeur** : un réseau à 3 couches cachées de 16 neurones apprend à classer chaque pixel (x,y) d'une image (noir/blanc) ; les couches successives plient et déforment l'espace pour produire des frontières complexes avec peu d'unités. Un réseau profond est généralement plus compact qu'un réseau peu profond mais très large.

---

## Liens avec l'analyse de données

### MLP et ACP (PCA)

- Un **auto-encodeur** (réseau auto-associatif) est entraîné à reproduire son entrée : `d = x`
- Architecture : entrée n → **couche de codage** (peu d'unités) → sortie n
- L'**encodeur** compresse, le **décodeur** reconstruit
- Avec des sorties **linéaires**, il réalise une transformation de **Karhunen-Loève** : il est équivalent à une **ACP** - la couche cachée engendre le même sous-espace que les premières composantes principales.

Avec des activations **non linéaires**, l'auto-encodeur capte des structures que l'ACP linéaire ne voit pas (sur MNIST ou des données textuelles, les classes sont mieux séparées dans l'espace du codage que sur les 2 premières composantes).

### MLP et LDA (analyse discriminante)

Un MLP entraîné pour classer effectue une analyse discriminante ; à chaque couche cachée, les classes sont de plus en plus séparées (diapo 47 : entrée → cachée 1 → cachée 2 → sortie, les nuages de classes se séparent progressivement).

---

## Réseaux structurés

Idée générale : **contraindre l'architecture** (connexions locales, poids partagés, récurrence) pour réduire le nombre de paramètres et injecter de la connaissance sur le problème.

### Réseau complètement connecté (diapo 48)

Chaque unité d'une couche est reliée à toutes celles de la suivante. Beaucoup de paramètres, aucune structure imposée.

### Réseaux récurrents simples avec contexte

- **Elman** : les sorties de la couche **cachée** à l'instant t−1 sont recopiées dans des **unités de contexte** `c(t−1)` qui servent d'entrées supplémentaires à t. Mémoire du passé interne.
- **Jordan** : ce sont les **sorties du réseau** `y(t−1)` qui sont réinjectées en contexte.
- **NETtalk** : synthèse de parole (texte → phonèmes). Le réseau regarde une **fenêtre glissante de lettres** (la lettre centrale + son contexte) et prédit le phonème de la lettre centrale.

### Réseaux récurrents (RNN) (diapos 52–54)

Un RNN a une **boucle** : l'information persiste d'un pas de temps au suivant.

$h_t = f_W( h_{t−1}, x_t )$ -> (nouvel état = fonction de l'ancien état et de l'entrée)
$y_t = f( h_t )$ -> (sortie au temps t)

Mêmes poids W à chaque pas de temps. **Dépliage (unrolling)** : un RNN = plusieurs copies du même réseau, chacune transmettant un message à la suivante.

**Avantages** :
- traite des **séquences de longueur variable** (contrairement aux réseaux classiques (comme le MLP) qui exigent une taille d'entrée fixe, un RNN traite les éléments de la séquence un par un, qu'elle fasse 3 ou 100 mots);
- la complexité (nb de paramètres) ne dépend pas de la longueur, il est fixe ;
- poids réutilisés : même fonction et les mêmes poids à chaque pas de temps

**Inconvénient majeur** :
- La multiplication en chaîne (effet multiplicatif) : Pour faire apprendre un RNN, l'erreur doit être rétrofusée à travers le temps (Backpropagation Through Time). Mathématiquement, cela revient à multiplier à répétition les mêmes matrices de poids et de dérivées (comme la fonction tanh) pour chaque pas de temps de la séquence.   
- le gradient évanescent : Si les valeurs de ces matrices sont inférieures à 1, le fait de les multiplier en boucle fait que le gradient devient presque égal à zéro au bout de quelques pas. Le réseau devient incapable d'apprendre des dépendances à long terme
- le gradient explosif : À l'inverse, si les valeurs sont supérieures à 1, les multiplications répétées font croître le gradient de façon exponentielle jusqu'à l'infini, ce qui fait planter l'algorithme (les valeurs deviennent NaN)

### LSTM et GRU

Deux cellules « à portes », qui traitent l'information de façon **additive** : **LSTM** et **GRU**. Une **porte sigmoïde** sort des nombres entre 0 et 1 : la proportion de chaque composante qui « passe » (0 = rien, 1 = tout). Ex. `f_t = σ(W_f·[h_{t−1}, x_t] + b_f)`.

Le cœur du LSTM est l'**état de cellule C\_t** : mémoire à long terme, qui traverse les pas de temps avec peu d'interactions linéaires, d'où un **flux de gradient ininterrompu** (les erreurs passées continuent d'influencer l'apprentissage). Trois portes :

- Porte d'oubli (forget) :  $f_t = σ( W_f·[h_{t−1}, x_t] + b_f )$     → quelle part de C_{t−1} on garde

- Porte d'entrée (input) :  $i_t = σ( W_i·[h_{t−1}, x_t] + b_i )$     → quelle part de la nouveauté on accepte

- Candidat mémoire       :  $C̃_t = tanh( W_C·[h_{t−1}, x_t] + b_C )$ → crée une nouvelle information potentielle à ajouter

- Mise à jour de l'état  :  $C_t = f_t ∗ C_{t−1} + i_t ∗ C̃_t$           (∗ = produit terme à terme) → le plus important : on combine ce qu'on a gardé du passé et accepté du présent par addition

- Porte de sortie        :  $o_t = σ( W_o·[h_{t−1}, x_t] + b_o )$ → filtre l'état de la cellule mis à jour pour décider quelle partie de la mémoire sera affichée comme sortie

- Sortie                 :  $h_t = o_t ∗ tanh( C_t )$

La mise à jour **additive** de C\_t est la clé contre le gradient évanescent.

### Connexions locales et poids partagés (diapos 58–60)

- **Connexions locales** : chaque unité cachée ne « voit » qu'un **champ récepteur** (petite zone de l'entrée), ce qui réduit fortement le nombre de poids.
- **Poids partagés (contraints)** : le **même filtre** (noyau de convolution) est appliqué à toutes les positions → des **extracteurs de caractéristiques** (feature extractors) qui détectent la même caractéristique locale partout. Très peu de paramètres libres.
- **TDNN** (Time Delay Neural Network) : même principe appliqué à l'axe **temps** pour la reconnaissance de phonèmes sur un spectrogramme (fréquence × temps) ; le nombre de positions est environ `(N − d)/s + 1` pour une fenêtre d, un pas s.

---

## Réseaux convolutifs (CNN)

### Architecture type (LeNet)

Entrée (32×32) → **convolutions** (cartes de caractéristiques 6@28×28) → **sous-échantillonnage / pooling** (6@14×14) → convolutions (16@10×10) → pooling (16@5×5) → couches **entièrement connectées** (120, 84) → sortie (10 classes). Alternance détection de motifs / réduction de résolution.

### La convolution discrète

Transformation **linéaire**, qui préserve l'ordre (la structure spatiale), **parcimonieuse** (quelques entrées seulement contribuent à chaque sortie) et qui **réutilise les paramètres** (mêmes poids en toutes positions).

Calcul : on **fait glisser** le noyau sur la carte d'entrée ; à chaque position, on multiplie terme à terme noyau et zone recouverte, puis on **somme**. On décale ensuite le noyau et on recommence.

Définitions : **stride (pas)** = distance entre deux positions consécutives du noyau ; **zero padding** = nombre de zéros ajoutés au début et à la fin de chaque axe.

### Formules de taille de sortie

Notations : i = taille d'entrée, k = taille du noyau, s = stride, p = padding, o = taille de sortie (entrées/noyaux carrés).

| Relation | Cas | Formule | Exemple |
| --- | --- | --- | --- |
| 1 | pas de padding, s=1 | o = (i−k)+1 | i=4, k=3 → o=2 |
| 2 | padding p, s=1 | o = (i−k)+2p+1 | i=5, k=4, p=2 → o=6 |
| 3 | **half/same** : k impair, p=⌊k/2⌋, s=1 | o = i | i=5, k=3, p=1 → o=5 |
| 4 | **full** : p=k−1, s=1 | o = i + (k−1) | i=5, k=3, p=2 → o=7 |
| 5 | p=0, stride s | o = ⌊(i−k)/s⌋+1 |  |
| 6 | **général** | **o = ⌊(i+2p−k)/s⌋ + 1** |  |

À retenir : la relation 6 contient toutes les autres. Le plancher ⌊·⌋ implique que **plusieurs tailles d'entrée peuvent donner la même sortie** quand s>1 (certaines lignes/colonnes en bordure ne sont pas couvertes par le noyau).

### Pooling

On fait glisser une fenêtre et on applique une fonction de **résumé** (comme une convolution, mais non linéaire / pas de noyau appris) :

- **Max pooling** : le maximum de la fenêtre.
- **Average pooling** : la moyenne.
- Fenêtre 2×2 courante : élimine **3/4 des activations**.
- Rôle : réduire la résolution de la carte tout en gardant les caractéristiques utiles → **invariance aux petites translations** (et rotations légères), moins de calculs, moins de sur-apprentissage.

Exemple (diapo 77) : entrée 4×4 `[12 20 30 0 / 8 12 2 0 / 34 70 37 4 / 112 100 25 12]`, fenêtre 2×2, stride 2 :

- max pooling → `[20 30 / 112 37]` ;
- average pooling → `[13 8 / 79 20]` (ex. (12+20+8+12)/4 = 13 ; (37+4+25+12)/4 = 19,5 ≈ 20).

Variantes :
- **stochastique** (une activation tirée au hasard dans chaque région, effet de régularisation)
- **avec recouvrement** (fenêtres qui se chevauchent)
- **fractionnaire** (le taux de réduction peut être non entier, ce qui permet des réseaux plus profonds).

### Convolution transposée (« déconvolution »)

Opération qui va **dans le sens inverse** d'une convolution : elle part d'une carte ayant la forme de la sortie d'une convolution et reconstruit une carte ayant la forme de son entrée, avec un motif de connexions compatible.

Utilisations : On la retrouve dans le **décodage des auto-encodeurs convolutifs**, les tâches de segmentation d'images (pour retrouver la taille d'origine de l'image pixel par pixel) et la **super-résolution** (passer d'une image floue/basse résolution à une image haute résolution)

**Représentation matricielle** (comment relier la convolution classique et sa version transposée):
- **En convolution normale** : Si on aplatit l'entrée en un vecteur de taille $16$ (image $4 \times 4$) et la sortie en un vecteur de taille $4$ (image $2 \times 2$), l'opération s'écrit $y = C \cdot x$, où $C$ est une matrice creuse de taille $4 \times 16$.
- **En rétropropagation** : Pour faire remonter l'erreur dans l'autre sens lors de l'apprentissage d'une convolution normale, on multiplie par la matrice transposée $C^T$ (de taille $16 \times 4$), ce qui ramène l'information de $4$ à $16$ dimensions.
- **La convolution transposée** reprend exactement cette matrice $C^T$ pour l'utiliser comme une opération directe : on passe d'une petite dimension à une grande dimension en utilisant les mêmes poids $w$.

**Relation 8** : une convolution (s=1, p=0, noyau k) a pour transposée une convolution avec k'=k, s'=s, p'=k−1, et taille de sortie `o' = i' + (k−1)`. Elle équivaut à une convolution **full padding**. Exemple : convolution 4×4 → 2×2 ; sa transposée : 2×2 → 4×4.

### Exemples applicatifs du cours

- **LeNet** : reconnaissance de chiffres manuscrits (codes postaux) par rétropropagation sur un réseau convolutif.
- **Identification du locuteur** (Bennani & Gallinari 1991) : un TDNN extrait des caractéristiques sur le spectrogramme ; un **HMM par locuteur** utilise ces caractéristiques.
- **Segmentation médicale** : encodeur convolutif (Convolution + Batch Normalization + ReLU, pooling) puis décodeur (déconvolution, 1×1 convolution) → carte de segmentation de même taille que l'image.
- **Réseau « ver nématode »** : un réseau inspiré du système nerveux du ver pilote une voiture avec seulement 19 neurones de contrôle, après des couches convolutives de perception.

### Super-résolution (SRCNN / FSRCNN) : comptage des paramètres

Objectif : passer d'une image basse résolution (LR) à haute résolution (HR).

**SRCNN** (3 couches de convolution) :

1. Extraction de patchs : 64 filtres 1×9×9, ReLU → poids `1×9×9×64 = 5184` + 64 biais.
2. Mapping non linéaire : 32 filtres 64×1×1, ReLU → `64×1×1×32 = 2048` + 32 biais.
3. Reconstruction : 1 filtre 32×5×5, identité → `32×5×5×1 = 800` + 1 biais.

*Règle générale :* nombre de poids d'une couche convolutive = `(nb filtres) × (canaux d'entrée) × (hauteur) × (largeur)` ; biais = nb de filtres.

**FSRCNN** (plus compact, 5 étapes) : (1) *extraction* (56 filtres 1×5×5 : 1400 poids) ; (2) *shrinking* (12 filtres 56×1×1 : 672 poids) qui réduit la dimension des vecteurs de caractéristiques ; (3) *mapping non linéaire* (4 couches de 12 filtres 12×3×3 : 4×12×3×3×12 = 5184 poids) ; (4) *expanding* (56 filtres 12×1×1 : 672 poids) qui fait l'opération inverse du shrinking ; (5) *déconvolution* (un filtre 56×9×9 : 4536 poids) qui produit l'image HR. Total ≈ **12 464 poids** (+ quelques paramètres de PReLU).

---

## Deep learning : objectifs et difficultés

**Objectif** : **construire un nouvel espace de représentation** par des architectures profondes (comme les couches cachées d'un MLP, mais en plusieurs niveaux d'abstraction). L'apprentissage non supervisé entre les couches peut décomposer le problème en sous-problèmes distribués, de plus en plus abstraits.

**Difficultés de l'apprentissage supervisé de réseaux profonds :**

- Les premières couches sont mal apprises :
  - **diffusion du gradient** : l'erreur diminue en se propageant vers les couches précédentes
  - donc apprentissage très lent
  - plus le réseau est profond, plus la non-linéarité est forte, ce qui augmente les risques d'obstacles à l'optimisation
  - les couches basses restent avec des transformations peu utiles
- **Pas assez de données étiquetées** : on cherche des approches non supervisées, semi-supervisées ou par **transfert** pour exploiter les données non étiquetées.
- Davantage de **minima locaux** que dans les réseaux peu profonds.

**Réponses** : ReLU (gradient mieux propagé), bonne initialisation et normalisation des données, cellules à portes (LSTM/GRU), poids partagés (CNN), pré-apprentissage non supervisé (auto-encodeurs).

---

## Checklist de révision

- Expliquer le **problème d'affectation du crédit** et comment la GBP le résout.
- Écrire de mémoire : la sortie d'un MLP à 1 couche cachée ; le risque empirique ; la règle de descente de gradient.
- **Démontrer** δ\_j (sortie) = f'(a\_j)(y\_j−d\_j) et δ\_j (cachée) = f'(a\_j)Σ w\_hj δ\_h, puis Δw\_ji = −ε δ\_j s\_i.
- Dérivées : sigmoïde f(1−f), tanh 1−f², ReLU 0/1.
- Comparer sigmoïde / tanh / ReLU / variantes ; expliquer le *dying ReLU*.
- Justifier centrage-réduction et la règle d'initialisation en 1/√|Amont|.
- Différencier Elman / Jordan / RNN / LSTM / GRU ; écrire les équations du LSTM.
- Calculer la taille de sortie d'une convolution (relation 6) et faire une convolution et un pooling à la main.
- Expliquer convolution transposée (matrice C et Cᵀ, Relation 8).
- Citer Kolmogorov, Hornik (existentiel !), liens auto-encodeur–ACP et MLP–LDA.
- Lister les difficultés des réseaux profonds.

---



# PARTIE III du cours : représentations non supervisées, modèles génératifs, optimisation et généralisation

*Les premières diapos de la Partie III reprennent la convolution, le pooling et LeNet / TDNN déjà détaillés en Partie II (sections 13–14). Elles rappellent aussi l'objectif du deep learning et ses difficultés (Partie II, section 15) : c'est le point de départ de ce qui suit. **Les formules marquées « à vérifier » sont des formules standard que j'ai ajoutées parce que la diapo ne les montrait pas lisiblement.***

## III.1 Apprentissage non supervisé de représentations : l'auto-encodeur (diapos 4–13)

Rappel du problème : les premières couches d'un réseau profond sont mal apprises par rétropropagation (gradient qui se diffuse) et on manque de données étiquetées. Idée : **exploiter les données non étiquetées** pour apprendre de bonnes représentations, puis affiner.

**Auto-encodeur** (aussi *autoassociateur* ou réseau *Diabolo*) : réseau de neurones **non supervisé** qui apprend à **reproduire son entrée en sortie** (fonction identité) en passant par une **couche intermédiaire plus petite**. But : apprendre un **codage efficace** (compression, représentation distribuée) pour la **réduction de dimension**. Il est lié au concept de **codage parcimonieux** (*sparse coding*).

```
Encodeur :   Y  = f_θ(X)              (code latent, de petite taille)
Décodeur :   X̂ = g_θ'(Y)             (reconstruction de X)
Perte (à vérifier) :  L(X, X̂) = || X − X̂ ||² = Σ_i ( x_i − x̂_i )²
```

La perte est la **différence entre l'entrée réelle et l'entrée reconstruite** (erreur quadratique si l'entrée est réelle). Variante vue sur la diapo : poids du décodeur **liés** à ceux de l'encodeur, **W** (encodeur) et **Wᵀ** (décodeur), ce qui divise le nombre de paramètres par 2.

- **Comparaison avec l'ACP** : un auto-encodeur à sorties linéaires est équivalent à une ACP (Partie II, section 12). Avec des activations non linéaires, la **reconstruction est meilleure** que celle de l'ACP à dimension de code égale (diapo : images de chiens, ACP floue vs auto-encodeur net) et les **codes 2D** produits séparent mieux les classes (MNIST, base d'articles de presse).

## III.2 Auto-encodeurs empilés (SAE, Stacked AutoEncoders) (diapos 14–20)

On **empile** plusieurs auto-encodeurs pour former un réseau profond :

1. On entraîne un premier auto-encodeur sur les données X (non étiquetées) → il produit un **nouveau code** Y¹ de X.
2. On **jette le décodeur** et on utilise Y¹ comme **entrée** d'un second auto-encodeur, entraîné à reconstruire Y¹ → code Y². On répète : la couche cachée de l'auto-encodeur *i* devient l'entrée de l'auto-encodeur *i+1*.
3. Apprentissage **couche par couche** (*layer-wise*), non supervisé : chaque couche minimise l'erreur de reconstruction de **sa** entrée.
4. Une fois toutes les couches entraînées : deuxième phase appelée **fine-tuning** (réglage fin) : on empile les encodeurs, on ajoute une couche de classification et on entraîne **tout le réseau par rétropropagation supervisée**.

Intérêt : le pré-entraînement initialise les poids dans une région favorable de l'espace des paramètres, ce qui réduit le problème du gradient diffusé et exploite les données non étiquetées.

## III.3 Apprentissage auto-supervisé (diapos 21–26)

**Idée** : obtenir des étiquettes **gratuitement** à partir des données elles-mêmes. On construit une tâche supervisée où l'on **prédit une partie de l'information à partir du reste** : entrées et labels sont tous deux fournis par les données. (Réf. : Lilian Weng, 2019.)

- **Motivation** : produire un jeu de données propre et étiqueté coûte cher, alors que les données non étiquetées sont produites en continu.
- **Tâche prétexte (*pretext task*)** : la tâche inventée qui fournit la perte supervisée. On ne s'intéresse **pas** à sa performance finale, mais à la **représentation intermédiaire** apprise, qui doit porter un sens sémantique/structurel et servir à des **tâches aval (*downstream*)**.
- Évaluation classique : entraîner sur une tâche prétexte avec des images sans labels, puis brancher une couche intermédiaire sur une régression logistique multinomiale (ImageNet) : la précision mesure la qualité de la représentation.

**Exemples de tâches prétextes :**

| Famille | Principe |
| --- | --- |
| **Distorsion** (Exemplar-CNN, Dosovitskiy 2015) | Extraire N patchs 32×32 dans des zones à forts gradients (contours), appliquer des transformations aléatoires (translation, rotation, échelle) ; tous les patchs issus du même patch original forment **une même classe substitut** ; on apprend à les discriminer. |
| **Rotation** (Gidaris 2018) | Tourner l'image d'un multiple de 90° (0°, 90°, 180°, 270°) ; classification à **4 classes** : prédire la rotation appliquée. |
| **Patchs** (Doersch 2015) | Extraire deux patchs d'une image et prédire leur **position relative** : oblige à comprendre le contexte spatial des objets. |
| **Génératif** | Reconstruire l'entrée tout en apprenant un latent utile : *context encoder* (Pathak 2016, reconstruire une zone masquée), *split-brain autoencoder* (Zhang 2017, prédire une partie des canaux couleur à partir des autres). |

Les modèles génératifs sont tous auto-supervisés au sens large, mais l'**objectif** diffère : les génératifs visent des images diverses et réalistes ; la représentation auto-supervisée vise des **caractéristiques utiles à de nombreuses tâches**.

**Auto-apprentissage (*self-training*)** : un classifieur existant **F\_pl** (le « pseudo-étiqueteur ») prédit des **pseudo-étiquettes** sur un grand jeu non étiqueté, puis on **réentraîne** un nouveau modèle avec ces pseudo-étiquettes.

## III.4 Réseaux antagonistes génératifs (GAN) (diapos 27–47)

### Pourquoi des modèles génératifs ?

- **Modèle discriminatif** : étant donné une image X, prédit un label Y, i.e. estime **P(Y|X)**. Limites : il ne modélise pas **P(X)**, donc ne peut pas **générer** de nouvelles images.
- **Modèle génératif** : apprend la probabilité conjointe **P(X,Y)** (ou P(X)) ; peut comprendre la structure sous-jacente des données **même sans labels**, et **échantillonner**.

### Principe (Goodfellow et al., 2014)

GAN = **G**enerative **A**dversarial **N**etworks : un système de **deux réseaux en compétition** dans un **jeu à somme nulle**.

- **Générateur G** : reçoit un bruit gaussien z (distribution fixe) et produit un faux échantillon G(z). But : **tromper D**.
- **Discriminateur D** : classifieur qui estime la probabilité **D(x)** que x soit un échantillon **réel**. But : distinguer vrai et faux.

Au terme de l'apprentissage, G produit des échantillons que D ne sait plus distinguer des vrais (chambres à coucher réalistes, visages StyleGAN…).

### Entraînement du discriminateur

D est un classifieur binaire à sortie **sigmoïde**, entraîné par **entropie croisée binaire** sur N points réels x₁…x\_N et N points générés G(z₁)…G(z\_N) :

```
L_D = − Σ_{n=1}^{N} [ log D(x_n) + log( 1 − D(G(z_n)) ) ]
Pour G fixé :  D* = argmin_D L_D
```

Le premier terme pousse D(x) vers 1 sur les vrais, le second pousse D(G(z)) vers 0 sur les faux. (Résultat classique, à vérifier : le discriminateur optimal vaut D\*(x) = p\_data(x) / (p\_data(x) + p\_G(x)).)

### Entraînement du générateur

La performance de G est évaluée **via D** : G **maximise la perte du discriminateur**.

```
Coût possible :  L_G = − L_D
Pour D fixé :    G* = argmax_G L_D = argmax_G − Σ_n log( 1 − D(G(z_n)) )
En pratique :    G* = argmax_G Σ_n log D(G(z_n))        (perte « non saturante »)
```

**Pourquoi remplacer la perte ?** Au début, G est faible et D(G(z)) ≪ 1. Le terme log(1−D(G(z))) est alors **presque plat** près de 0 (gradient faible), tandis que **log D(G(z)) a une forte pente près de zéro** : le gradient est grand, ce qui **relance l'apprentissage** du générateur.

Formulation globale : **min\_G max\_D L(D, G)** (jeu minimax).

### Code PyTorch du cours (GAN sur des « double moons » 2D)

```python
z_dim = 32; hidden_dim = 128
net_G = nn.Sequential(nn.Linear(z_dim,hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,2))
net_D = nn.Sequential(nn.Linear(2,hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,1), nn.Sigmoid())
# boucle d'apprentissage (Adam, lr=1e-3, batch 50, 500 epochs)
z = torch.empty(batch_size, z_dim).normal_()     # bruit gaussien
fake_batch = net_G(z)
loss = -torch.mean(torch.log(1 - net_D(fake_batch)) + torch.log(net_D(real_batch)))   # amélioration de D
loss = -torch.mean(torch.log(net_D(net_G(z))))                                          # amélioration de G
```

À chaque mini-lot, on **alterne** : (1) une étape de gradient sur D avec L\_D ; (2) une étape sur G avec la perte non saturante. G produit des points en dimension 2 (les « lunes ») à partir d'un bruit en dimension 32.

### GAN convolutifs (DCGAN, Radford 2015)

Idées clés : remplacer les couches cachées complètement connectées par des **convolutions** ; **convolutions à pas fractionnaire (transposées)** dans le générateur ; **Batch Normalization** après chaque couche ; dans G : **ReLU** sur les couches cachées et **tanh** en sortie (images normalisées dans \[−1,1\]).

### Problèmes des GAN

- **Distribution implicite** : on ne peut pas calculer P(X) ; les GAN ne servent qu'à **échantillonner / générer**.
- **Entraînement difficile** : non-convergence, **mode collapse** (G ne produit que quelques types d'échantillons) ; D cherche à maximiser sa récompense, G à la minimiser ; **la SGD n'a pas été conçue pour trouver l'équilibre d'un jeu**, donc on peut ne jamais converger.

## III.5 Machines de Boltzmann restreintes (RBM) et DBN (diapos 48–64)

**RBM** : version stochastique d'un réseau de Hopfield (réseau récurrent) ; utilisée comme mémoire associative et pour la réduction de dimension. **Modèle génératif non supervisé à énergie** avec une couche de **neurones visibles binaires v** et une couche de **neurones cachés binaires h**. « **Restreinte** » : **pas de connexions à l'intérieur d'une même couche** (graphe biparti). Cas particulier d'un **Deep Belief Network (DBN)**.

### Énergie et probabilités

```
Énergie :  E(v,h) = − bᵀv − cᵀh − vᵀWh = − Σ_k b_k v_k − Σ_j c_j h_j − Σ_j Σ_k W_jk h_j v_k
Jointe  :  p(v,h) = (1/Z) exp( −E(v,h) ),     Z = Σ_v Σ_h exp( −E(v,h) )
Marginale: p(v) = Σ_h exp( −E(v,h) ) / Z
```

b = biais des unités visibles, c = biais des unités cachées, W = poids visible-caché. Faible énergie = configuration probable. Z est la **fonction de partition** (normalisation), très coûteuse à calculer.

### Conditionnelles factorisées (la propriété clé)

Dérivation de p(h|v) :

```
p(h|v) = p(v,h) / p(v) = (1/Z′) exp( cᵀh + vᵀWh )          (bᵀv ne dépend pas de h : absorbé dans la constante)
       = (1/Z′) ∏_j exp( c_j h_j + vᵀW_{:,j} h_j )          (exponentielle d'une somme = produit)
```

Donc, **sachant v, les h\_j sont conditionnellement indépendants**, et pour h\_j ∈ {0,1} :

```
P(h_j = 1 | v) = exp(c_j + vᵀW_{:,j}) / ( exp(0) + exp(c_j + vᵀW_{:,j}) ) = sigmoïde( c_j + vᵀW_{:,j} )
P(h | v) = ∏_j sigmoïde( c_j + vᵀW_{:,j} )        (peuvent servir de « caractéristiques »)
P(v | h) = ∏_i sigmoïde( b_i + W_{i,:} h )
```

Conséquence pratique : on peut **échantillonner toute une couche en parallèle**, ce qui rend les RBM utilisables.

### Apprentissage : exemple « apprendre des caractéristiques pour reconstruire des 2 » (diapos 59–61)

Image 16×16 pixels, 50 neurones caractéristiques binaires. Principe (règle de type *divergence contrastive*, nom à vérifier dans tes notes) :

- Avec la **donnée réelle** : **incrémenter** les poids entre un pixel actif et une caractéristique active.
- Avec la **reconstruction** (à partir des caractéristiques activées) : **décrémenter** les poids entre un pixel actif et une caractéristique active.

Résultat : 50 × 256 poids ; **chaque neurone apprend une caractéristique différente** ; la reconstruction est très proche des données (et même « meilleure que la réalité », plus propre). Sur des chiffres d'une **autre classe** (jamais vue), le réseau « essaie de voir chaque image comme un 2 » : il n'a appris que les caractéristiques des 2.

### Deep Belief Network (Hinton 2002 ; Smolensky 1986)

On **empile des RBM** : la couche cachée de l'un devient la couche visible du suivant. Première couche = RBM qui apprend des **contours** (*edges*) à partir des pixels ; couches supérieures (DBN) = **combinaisons de contours**, etc. : hiérarchie de représentations apprise **de façon non supervisée, couche par couche** (même logique que les SAE).

## III.6 Variantes de la rétropropagation : accélérer la descente de gradient (diapos 65–77)

Problème de la descente de gradient simple : le pas ε est difficile à régler (trop petit : lent ; trop grand : oscillations, surtout dans des « vallées » étroites). Les variantes ci-dessous adaptent la **direction** ou le **pas**. Notation : g(t) = ∂R̃/∂w(t) = gradient à l'instant t.

### 1) Momentum (Rumelhart, Hinton, Williams 1986)

Idée : donner une **inertie** aux poids pour **éviter les oscillations** ; momentum = descente de gradient + **lissage**.

```
Δw(t) = w(t+1) − w(t) = −ε · ∂R̃/∂w(t) + α · Δw(t−1),      0 ≤ α < 0,9
```

**Dérivation : l'ajustement est une moyenne pondérée de toutes les adaptations passées.** En développant Δw(t−1) = −ε g(t−1) + α Δw(t−2), etc. :

```
Δw(t) = −ε Σ_{k=0}^{∞} α^k · g(t−k)
```

**Effet d'accélération** : si le gradient est constant (g(t−k) = ∇) :

```
Δw(t) = −ε ∇ Σ_{k≥0} α^k = ( −ε / (1−α) ) ∇
```

Le **pas effectif** est ε̂ = ε/(1−α) : plus α est proche de 1, plus le pas est amplifié (ex. α = 0,9 → pas ×10) sur les directions de gradient stable. Dans les directions où le gradient **change de signe** (oscillations), les termes se compensent : les oscillations sont amorties.

### 2) Méthode de Vogl (« Bold Driver », 1988) : pas adaptatif global

```
ε(t) = α ε(t−1)   si R̃(w(t)) < R̃(w(t−1))              (l'erreur baisse → on accélère)
ε(t) = β ε(t−1)   si R̃(w(t)) > 1,05 · R̃(w(t−1))       (l'erreur monte de plus de 5 % → on freine)
ε(t) = ε(t−1)     sinon,        avec α > 1 et β < 1   (ex. α = 1,05 ; β = 0,7)
```

### 3) Silva & Almeida (1990) : un pas **par connexion**

```
w_ij(t+1) = w_ij(t) − ε_ij(t) · ∂R̃/∂w_ij(t)
ε_ij(t) = α ε_ij(t−1)  si  g_ij(t)·g_ij(t−1) > 0   (même signe : on accélère)
ε_ij(t) = β ε_ij(t−1)  si  g_ij(t)·g_ij(t−1) < 0   (signe opposé : on freine)
ε_ij(t) = ε_ij(t−1)    sinon          (ex. 1,1 < α < 1,3 ; β = 0,7)
```

### 4) Corrélation du gradient (Chan & Fallside, 1987)

On mesure le **cosinus de l'angle entre deux gradients successifs** :

```
cos(θ) = g(t−1)ᵀ g(t) / ( ||g(t−1)|| · ||g(t)|| )
ε(t) = ε(t−1) · β⁺ · cos(θ)   si cos(θ) > 0       (même direction : on accélère)
ε(t) = ε(t−1) · β⁻             sinon             (changement de direction : on freine)
β⁺ = 1,005 ;  β⁻ = 0,8
```

### 5) Quickprop (Fahlman, 1988)

```
Δw(t) = β · Δw(t−1),    β = g(t) / ( g(t−1) − g(t) )
```

Idée : approcher la **dérivée seconde** par la différence finie \[g(t) − g(t−1)\] / Δw(t−1) et sauter directement au minimum de la parabole locale. β > 0 ⇒ même direction ; β < 0 ⇒ on change de direction.

### 6) Rprop (Riedmiller & Braun, 1993) : propagation résiliente

Les adaptations ne dépendent **que du signe** du gradient, pas de sa valeur :

```
Δ_ij(t) = ε⁺ Δ_ij(t−1)  si g_ij(t)·g_ij(t−1) > 0     (0 < ε⁻ < 1 < ε⁺ ; ex. ε⁻ = 0,5 ; ε⁺ = 1,2)
Δ_ij(t) = ε⁻ Δ_ij(t−1)  si g_ij(t)·g_ij(t−1) < 0
Δ_ij(t) = Δ_ij(t−1)    sinon

Δw_ij(t) = −Δ_ij(t) si g_ij > 0 ;   +Δ_ij(t) si g_ij < 0 ;   0 sinon          (initialisation Δ_ij(0) = 0,001)
```

Avantage : insensible à l'**amplitude** du gradient (qui peut être minuscule dans les couches profondes).

## III.7 Règles du second ordre (diapos 78–86)

### Méthode de Newton

Développement de Taylor à l'ordre 2 de la fonction coût autour de ŵ :

```
R̃(w) ≈ R̃(ŵ) + (w−ŵ)ᵀ ∇R̃(ŵ) + ½ (w−ŵ)ᵀ H(ŵ) (w−ŵ)       avec le Hessien  H_ij = ∂²R̃ / ∂w_i ∂w_j
```

En dérivant : ∇R̃(w) = ∇R̃(ŵ) + H(ŵ)(w−ŵ). **Au minimum, ∇R̃(w) = 0**, donc ∇R̃(ŵ) + H(ŵ)(w−ŵ) = 0, d'où :

```
w(t+1) = w(t) − H⁻¹(w(t)) · ∇R̃(w(t))          (« direction de Newton »)
```

Convergence très rapide près d'un minimum (le Hessien tient compte de la **courbure**), mais méthode **impraticable** : calculer et inverser le Hessien (|w|×|w|) est **très coûteux** quand il y a des milliers de poids. D'où deux familles d'approximations :

| Famille | Méthode | Coût (diapo 79) |
| --- | --- | --- |
| **Recherche directionnelle** (line search) | plus grande pente ; gradients conjugués | O(\|w\|) |
| **Approximation du Hessien** | quasi-Newton | O(\|w\|²) |
|  | pseudo-Newton ; Levenberg-Marquardt | O(\|w\|) |

### Recherche directionnelle : méthode de plus grande pente

On ne calcule pas de dérivée seconde, mais on cherche à chaque instant le **meilleur point dans une direction donnée** :

```
w(t+1) = w(t) + α d_t,      α choisi pour minimiser R̃(w(t) + α d_t)      (recherche linéaire)
Plus grande pente :  d_t = −∇R̃(w(t))
```

### Gradients conjugués

Le défaut de la plus grande pente : après une recherche linéaire exacte, deux directions successives sont **orthogonales** (zigzag). Les **directions conjuguées** évitent de défaire le travail précédent : d\_{t+1} est conjuguée à d\_t si la composante du gradient parallèle à d\_t, qui vient d'être annulée, **reste nulle** quand on se déplace le long de d\_{t+1}. Formellement : d\_tᵀ H d\_j = 0 pour i ≠ j.

```
d_{t+1} = − g_{t+1} + β_t d_t        (nouvelle direction = direction du gradient + compromis avec l'ancienne)
```

Algorithme : (1) choisir w(0) ; (2) g₀ = ∇R̃(w(0)), d₀ = −g₀ ; (3) à l'itération t, α\_t = argmin\_α R̃(w(t) + α d\_t) et w(t+1) = w(t) + α\_t d\_t ; (4) tester le critère d'arrêt ; (5) calculer le nouveau gradient g\_{t+1} ; (6) calculer β\_t ; (7) d\_{t+1} = −g\_{t+1} + β\_t d\_t ; (8) t ← t+1, retour en 3.

Règles pour β\_t (g = gradient) :

```
Hestenes–Stiefel :  β_t = g_{t+1}ᵀ (g_{t+1} − g_t) / ( d_tᵀ (g_{t+1} − g_t) )
Polak–Ribière   :  β_t = g_{t+1}ᵀ (g_{t+1} − g_t) / ( g_tᵀ g_t )
Fletcher–Reeves  :  β_t = g_{t+1}ᵀ g_{t+1} / ( g_tᵀ g_t )
```

### Quasi-Newton (BFGS)

On construit une approximation H′ ≈ H⁻¹ **sans calculer le Hessien** : w(t+1) = w(t) − ε H′(t) ∇R̃(w(t)), où ε est choisi par recherche linéaire, puis H′ est **mise à jour** à partir des seuls gradients. Règle **BFGS** (Broyden-Fletcher-Goldfarb-Shanno) :

```
p = w(t+1) − w(t)          v = ∇R̃(w(t+1)) − ∇R̃(w(t))          u = p/(pᵀv) − H′v/(vᵀH′v)
H′(t+1) = H′ + p pᵀ/(pᵀv) − H′ v vᵀ H′/(vᵀ H′ v) + (vᵀ H′ v) u uᵀ
```

### Pseudo-Newton (Becker & Le Cun, 1988) et Levenberg-Marquardt

On n'utilise qu'une **approximation diagonale** du Hessien :

```
Pseudo-Newton :  Δw_ij = − ∇R̃(w_ij) / ( |σ_ij| + μ ),    σ_ij = x_j² γ_i,   γ_i = ∂δ_i / ∂a_i     (μ évite la division par 0)
Levenberg–Marquardt :  Δw_ij = − ∇R̃(w_ij) / σ_ij,   σ_ij = (1+λ) H_ii     (approximation du Hessien défini positif)
```

Ici le gradient ∇R̃(w\_ij) = δ\_i x\_j (voir rétropropagation) : chaque poids est mis à l'échelle par **sa propre courbure**.

---

## III.8 Apprentissage et généralisation (diapos 87–103)

Plan : formalisme ; analyse comportementale (**dilemme biais-variance**) ; **dimension VC** ; ERM ; cohérence de l'ERM ; **SRM** ; heuristiques (régularisation formelle, bruit en entrée, arrêt précoce, régularisation structurelle).

### Formalisme

Données D\_N = {(x(n), d(n))}, tirées de p(x,d) = p(x) p(d|x). On cherche dans une famille F = {ψ(x,w), w ∈ Ω} la fonction minimisant le **risque théorique** (erreur de généralisation) :

```
R(w) = ∫ L[ d, ψ(x,w) ] dp(x,d),      ψ(x,w*) = argmin_{ψ} R(w)
MSE :  R(w) = ∫ [ d − ψ(x,w) ]² dp(x,d)
```

### Analyse comportementale : décomposition de l'erreur

On estime ψ(x,w) à partir de N réalisations D\_N du couple (X,D). Pour la MSE :

```
R(w) = E[ (D − ψ(X,w))² ]
     = E[ (D − E[D|X])² ]  +  E[ (E[D|X] − ψ(X,w))² ]
          (constante : erreur bayésienne,       (minimiser R(w) équivaut à minimiser ce terme)
           liée au bruit du problème)
```

La **solution optimale** ψ(x,w\*) = **E\[D|X\]** : c'est la meilleure approximation au sens des moindres carrés de l'espérance conditionnelle.

En faisant apparaître la dépendance de l'estimateur à l'échantillon (ψ\_{D\_N}), l'erreur se décompose en **trois termes** :

```
R = E[(D − E[D|X])²]                              → erreur bayésienne (incompressible)
  + E_X[ ( E[D|X] − E_{D_N}[ψ_{D_N}(X,w)] )² ]      → (BIAIS)² : écart moyen entre l'estimateur et la solution optimale
  + E[ ( ψ_{D_N}(X,w) − E_{D_N}[ψ_{D_N}(X,w)] )² ] → VARIANCE : sensibilité de l'estimateur à l'échantillon D_N
```

### Dilemme biais-variance (Geman, Bienenstock, Doursat 1992)

Un bon estimateur = **bonne précision (faible biais) + bonne stabilité (faible variance)**. Quand la **complexité du modèle augmente**, le biais² **diminue** (le modèle peut représenter plus de choses) mais la variance **augmente** (il s'adapte aux particularités de l'échantillon). L'erreur globale a un **minimum** à une complexité intermédiaire : il faut un **compromis**.

Exemple de la diapo 92 : points tirés d'une courbe en bosse (bleue) avec du bruit. Droite verte = **biais élevé** (sous-apprentissage, modèle trop simple) ; courbe rouge qui passe par tous les points = **variance élevée** (sur-apprentissage, modèle trop complexe).

### Dimension de Vapnik-Chervonenkis (VC-dim)

**Définition** : mesure de la **capacité** (complexité, pouvoir d'expression) d'un ensemble de fonctions pour la classification binaire. C'est le **cardinal du plus grand ensemble de points que l'algorithme peut « pulvériser » (*shatter*)** :

```
VC-dim = max { k : ∃ S, card(S) = k, F pulvérise S }
```

Pulvériser S = réaliser **toutes les étiquetages possibles** (2^k) des points de S.

- **Une droite dans ℝ² a VC-dim = 3** : elle sépare toutes les étiquetages de 3 points non alignés (8 cas), mais **pas** 4 points (cas de type XOR). Plus généralement, un **hyperplan de ℝⁿ a VC-dim = n+1**.
- Connue pour les modèles simples ; pour les modèles complexes (MLP) on n'a que des **bornes**.

**MLP** (Baum & Haussler 1989) :

- avec M unités à seuil et W poids : **VC-dim ≤ 2 W log₂(e M)**. Si l'erreur d'apprentissage vaut ε/2 et qu'on veut E\_test < ε, il faut **N ≥ O( (W/ε) log₂(M/ε) )** exemples ;
- avec E entrées et C unités cachées : **VC-dim ≥ 2 ⌊C/2⌋ E**.

À retenir : **le nombre d'exemples nécessaires croît avec le nombre de poids** (règle empirique : N ≈ W/ε).

### Minimisation du risque empirique (ERM) et sa cohérence

Le risque théorique n'est pas calculable (p(x,y) inconnue), donc on applique un **principe d'induction** :

```
R̃(w) = (1/N) Σ_n L[ d(n), ψ(x(n),w) ],        ψ(x,w⁺) = argmin R̃(w)
```

**Question** : le minimiseur empirique ψ(x,w⁺) est-il proche de l'optimum ψ(x,w\*) quand N → ∞ ? **Théorème (Vapnik & Chervonenkis, 1971)** : l'ERM est **cohérent** ssi R̃(w) converge **uniformément** vers R(w) sur F :

```
lim_{N→∞} P{ sup_{w∈Ω} | R(w) − R̃(w) | > ε } = 0,   ∀ ε > 0
```

**Intervalle de confiance** : avec probabilité 1 − δ, pour N observations, pour tout w :

```
R̃(w) − ε(N, VC-dim(w), δ)  ≤  R(w)  ≤  R̃(w) + ε(N, VC-dim(w), δ)
```

Pour un classifieur ψ ∈ {0,1} :

```
ε(N, VC, δ) = √( (VC/N) [ 1 + log( 2N / VC ) ] − (1/N) log( δ/4 ) )
```

Lecture : ε (largeur de l'intervalle) **croît avec la VC-dim** et **décroît avec N**. R̃(w) **diminue** avec la capacité du modèle, mais ε **augmente** : le risque garanti R̃ + ε passe par un minimum ; au-delà, c'est le **sur-apprentissage** (*over-fitting*).

### Minimisation du risque structurel (SRM, Vapnik)

L'ERM est efficace quand N est grand ou le problème simple. Principe du SRM : minimiser le **risque garanti**

```
R_garanti(w) = R̃(w) + ε(N, VC-dim(w), δ)
```

On définit une suite de classes de modèles **emboîtées** : S₁ ⊂ S₂ ⊂ … ⊂ S\_n, avec VC-dim(S₁) < VC-dim(S₂) < … < VC-dim(S\_n) (ex. réseaux avec de plus en plus d'unités cachées). **Deux étapes** :

1. Dans chaque S\_i, calculer le minimum de l'erreur empirique : R̃(w\_i\*) = min\_{w∈Ω\_i} R̃(w).
2. Choisir la classe S\* qui minimise le risque garanti : S\* = argmin\_{S\_i} \[ R̃(w\_i\*) + ε(N, VC-dim(S\_i), δ) \].

### Heuristiques pour ajuster la capacité de généralisation

Problème clé : **ajuster la capacité du système au problème et aux données** pour obtenir la **meilleure généralisation**. But : systèmes opérationnels, méthodes heuristiques. Méthode : **restreindre l'espace de recherche** de la solution ; **régulariser**.

---

## III.9 Régularisation (diapos 104–120)

**Définition** : ensemble de techniques qui visent à **réduire l'erreur de généralisation sans trop dégrader l'erreur d'apprentissage**.

**Causes du sur-apprentissage** : (1) modèle assez complexe pour **modéliser le bruit** des données d'apprentissage ; (2) jeu d'apprentissage **trop petit** pour représenter correctement la distribution sous-jacente. Un modèle trop simple fait du **sous-apprentissage** ; la complexité idéale ne se déduit pas des données → on **contraint** un modèle complexe.

**Trois catégories** (efficacité entre parenthèses) :

- **Modifier la fonction de perte** : L2 (forte), L1 (forte), régularisation par entropie (forte).
- **Modifier l'échantillonnage** : augmentation de données (faible), validation croisée k-fold (moyenne).
- **Modifier l'algorithme d'apprentissage** : dropout (forte), injection de bruit (faible).

### Régularisation L2 (ridge, « weight decay »)

Régression linéaire : f\_β(x) = β₀ + Σ\_i β\_i x\_i. On ajoute à la perte la **norme L2 des poids** :

```
min_β  Σ_p || y^p − f_β(x^p) ||² + λ || β ||₂²        λ ≥ 0
```

L'**hyperparamètre λ** règle le compromis sur/sous-apprentissage : λ grand → poids petits → modèle simple (sous-apprentissage) ; λ petit → sur-apprentissage possible. Interprétation géométrique (cas 2D) : la solution est le **point du contour d'erreur MSE (ellipses bleues) qui se trouve dans la boule L2 (un disque)** ; augmenter λ revient à **réduire le rayon de la boule**. Les poids deviennent très petits mais **jamais exactement nuls**.

### Régularisation L1 (lasso)

```
min_β  Σ_p || y^p − f_β(x^p) ||² + λ || β ||₁        λ ≥ 0       (||β||₁ = Σ |β_i|)
```

La boule unité L1 est un **losange avec des coins** sur les axes : l'optimum tombe souvent sur un coin, donc **certains poids deviennent exactement nuls** → modèle **parcimonieux** (sélection de variables). C'est l'avantage de L1 sur L2.

### Régularisation par entropie (classification)

La sortie est une distribution de probabilité f\_θ(x) = p(classe = c | x), avec Σ\_c f\_c(x) = 1. L'**entropie** mesure l'incertitude de la distribution (plus d'incertitude = plus d'entropie). On modifie la perte :

```
min_θ  loss(y, f_θ(x)) − λ · ( − Σ_c f_c(x) log f_c(x) )        λ ≥ 0
```

On veut que la distribution de sortie garde **un certain degré d'incertitude** (donc **augmenter l'entropie**) ; comme on minimise, on **multiplie par −1**. Plus λ est grand, plus la distribution de sortie est **uniforme** (le modèle est moins « sûr de lui », donc moins sur-ajusté).

### Augmentation de données (faible)

Agrandir artificiellement le jeu d'apprentissage par des transformations de l'entrée : recadrage aléatoire, rotation, dilatation, retournement, ajout d'un peu de bruit, changements de couleur… (exemple du papillon).

### Validation croisée k-fold (moyenne)

On découpe le jeu en **k sous-ensembles disjoints** ; on entraîne **k modèles** : pour chacun, un sous-ensemble sert à la validation et les k−1 autres à l'apprentissage. On moyenne les k scores : estimation plus fiable de l'erreur de généralisation (et aide à choisir les hyperparamètres comme λ).

### Dropout (forte)

Pour les réseaux de neurones : à chaque itération, des **connexions (ou unités) sont supprimées au hasard** selon un **taux de dropout** (% de connexions retirées), et le réseau restant est entraîné ; à l'itération suivante, d'autres sont retirées. Cela empêche les unités de se **co-adapter** et donne une meilleure généralisation. Le taux doit être choisi avec soin ; **valeur conseillée : entre 0,25 et 0,4**. (Exemple de la diapo : taux de 30 % → 11 connexions actives sur 16.)

### Injection de bruit (faible)

- **Bruit sur les poids** : à chaque mise à jour, ajouter un petit bruit aléatoire aux poids → caractéristiques plus robustes. Peu efficace comme régularisateur.
- **Bruit sur les entrées** (Bishop 1995 ; Matsuoka) : pendant l'apprentissage, on ajoute un bruit aléatoire ξ à chaque entrée, **la sortie désirée restant inchangée** : une petite zone de l'espace d'entrée autour de chaque exemple est associée à la même sortie → **lissage** de la fonction apprise autour des points d'apprentissage.

**Résultat important : apprendre avec du bruit en entrée équivaut à une régularisation de Tikhonov.** Dérivation :

```
R_ξ(w) = ½ Σ ∫∫∫ [ d − ψ(x+ξ, w) ]² p(d|x) dp(x,d) p(ξ) dξ

Développement de Taylor (ordre 2) :
ψ(x+ξ,w) = ψ(x,w) + Σ_i ξ_i ∂ψ/∂x_i + ½ Σ_i Σ_j ξ_i ξ_j ∂²ψ/∂x_i∂x_j + O(ξ³)

Hypothèses sur le bruit :  E(ξ) = 0  et  E(ξξᵀ) = λ I   (bruit centré, indépendant d'une composante à l'autre, de variance λ)
```

En remplaçant dans R\_ξ et en prenant l'espérance sur ξ, les termes d'ordre 1 s'annulent (E(ξ)=0), et E(ξ\_iξ\_j) = λ δ\_ij ne laisse que les termes diagonaux :

```
R_ξ(w) = R(w) + λ Ω
Ω = ½ Σ ∫∫ { ( ∂ψ/∂x_i )² + ½ (d − ψ) ∂²ψ/∂x_i² } p(d|x) dp(x,d)      (somme sur les exemples et sur les composantes i)
```

Le terme Ω pénalise les **grandes dérivées de ψ par rapport aux entrées** : on force la fonction à être **lisse** (peu sensible aux petites variations de x). C'est une régularisation, avec λ = variance du bruit.

### Arrêt précoce (*early stopping*)

**Arrêter l'apprentissage avant la convergence** : on suit l'erreur sur un jeu de **validation** à côté de l'erreur d'apprentissage. L'erreur d'apprentissage baisse toujours ; l'erreur de validation baisse puis **remonte** (sur-apprentissage). On **s'arrête au minimum de l'erreur de validation** (Bishop 1995). Effet proche d'une régularisation : en partant de poids petits, un arrêt précoce limite la croissance des poids, donc la complexité effective du modèle.

---

## III.10 Checklist de révision (Partie III)

- Expliquer l'**auto-encodeur**, sa perte de reconstruction, sa différence avec l'ACP ; décrire le **SAE** (pré-entraînement couche par couche + fine-tuning).
- Définir l'apprentissage **auto-supervisé**, la **tâche prétexte**, donner 3 exemples (rotation, patchs, Exemplar-CNN) ; expliquer le **self-training** (pseudo-labels).
- **GAN** : rôle de G et D ; écrire L\_D et la perte non saturante de G ; expliquer **pourquoi** on remplace log(1−D(G(z))) par log D(G(z)) ; citer les problèmes (mode collapse, non-convergence) ; DCGAN.
- **RBM** : énergie, p(v,h), Z ; **démontrer** que P(h\_j=1|v) = sigmoïde(c\_j + vᵀW\_{:,j}) ; différence avec un auto-encodeur ; DBN = RBM empilés.
- **Momentum** : formule, développement en série, pas effectif ε/(1−α).
- Citer et écrire : Bold Driver, Silva-Almeida, Quickprop, Rprop (règles de signe).
- **Newton** : dériver w(t+1) = w(t) − H⁻¹∇R ; pourquoi c'est coûteux ; gradients conjugués (β\_t de Fletcher-Reeves/Polak-Ribière) ; quasi-Newton BFGS ; pseudo-Newton / Levenberg-Marquardt (Hessien diagonal).
- **Biais-variance** : démontrer la décomposition en erreur bayésienne + biais² + variance ; interpréter la courbe en U.
- **VC-dim** : définition (pulvériser), droite dans ℝ² = 3, hyperplan = n+1, bornes pour le MLP.
- **ERM**, cohérence (convergence uniforme), intervalle de confiance ε(N,VC,δ), **SRM** (2 étapes, classes emboîtées).
- Régularisation : L2 vs L1 (géométrie : disque vs losange ; L1 → parcimonie), entropie, augmentation de données, k-fold, **dropout** (0,25–0,4), bruit en entrée ≡ Tikhonov (dérivation), **arrêt précoce**.


---

# PARTIE IV du cours : prototypes et apprentissage non supervisé

*Objectifs du cours : (1) opposer apprentissage par poids et par dictionnaire, construire et entraîner un réseau RBF ; (2) classer avec des prototypes étiquetés (LVQ1, LVQ2.1, GLVQ, apprentissage de pertinence) ; (3) expliquer l'apprentissage non supervisé et relier Hebb, Oja et Sanger à l'ACP ; (4) exécuter k-means et la carte de Kohonen (SOM) à la main ; (5) comparer SOM, Neural Gas, GNG et ART, expliquer auto-encodeurs et auto-supervisé ; (6) choisir une méthode et l'évaluer sans étiquettes. **Les formules marquées « à vérifier » sont des compléments standard que je n'ai pas lus sur les diapos.***

## IV.1 Deux façons d'apprendre : poids ou dictionnaire (diapos 4–5)

|  | Modèles à **poids** (MLP, CNN, Transformers) | Modèles à **dictionnaire** (k-NN, RBF, LVQ, et k-means, SOM en partie B) |
| --- | --- | --- |
| Stocke | des millions de poids répartis sur des couches | un dictionnaire de **référents** : prototypes, centres, vecteurs de codage |
| Décide par | enchaînement de sommes pondérées et de non-linéarités | **similarité** de l'entrée aux référents |
| Apprend par | descente de gradient globale (rétropropagation) | placement des référents, souvent **local** |
| Lecture | difficile : connaissance **distribuée** | facile : chaque référent est un exemple typique |

Un spectre : **k-NN** (dictionnaire = tous les exemples) → **LVQ, RBF** (dictionnaire compact) → **MLP, CNN** (aucun dictionnaire, que des poids). Trois questions pour tout modèle à dictionnaire :

|  | k-NN | RBF | LVQ |
| --- | --- | --- | --- |
| Que contient le dictionnaire ? | tous les exemples | M centres (sans étiquette) | quelques prototypes étiquetés |
| Comment compare-t-on ? | une distance | une gaussienne de la distance | une distance (apprise dans ω-LVQ, GRLVQ) |
| Comment décide-t-on ? | vote des k plus proches | somme pondérée de similarités | classe du prototype le plus proche |
| Comment apprend-on ? | rien : on stocke | k-means puis moindres carrés | attirer / repousser les prototypes |

Mêmes idées, plusieurs noms : prototype = référent = centre = vecteur de codage (*codebook vector*) = *codeword* = vecteur de référence. Partie A : dictionnaires construits **avec** étiquettes ; Partie B : **sans** étiquettes.

---

## IV.2 Réseaux RBF : un dictionnaire de centres (diapos 6–22)

**Idée** : approcher toute fonction en **additionnant des « bosses » locales** (comme construire un paysage avec des tas de sable). Chaque unité cachée est un **détecteur local** : elle répond fortement près de son centre μ\_j et presque pas ailleurs.

```
Unité cachée :  φ_j(x) = exp( − ||x − μ_j||² / (2σ_j²) )        (gaussienne ; μ_j = centre/prototype, σ_j = largeur)
Sortie       :  y_k(x) = w_k0 + Σ_{j=1}^{M} w_kj · φ_j(x)           (combinaison linéaire des bosses)
```

Centres = **où** sont les bosses ; largeurs = **jusqu'où** elles portent ; poids = **hauteur** des bosses. Version à covariance : φ\_j(x) = exp( −½ (x−μ\_j)ᵀ Σ\_j⁻¹ (x−μ\_j) ).

### Interpolation exacte (Powell, 1987)

On place **une fonction non linéaire par point d'apprentissage** :

```
ψ(x,w) = Σ_{n=1}^{N} w(n) · φ( ||x − x(n)|| )     ⟹   Φ w = d     avec  Φ_{nn'} = φ( ||x(n) − x(n')|| )     ⟹   w = Φ⁻¹ d
```

Le modèle passe **exactement** par les N points (donc interpole aussi le bruit : sur-apprentissage). Fonctions radiales possibles : **gaussienne** exp(−x²/2σ²) ; **multiquadratique inverse** (x²+σ²)^(−α) ; **spline plaque mince** x² ln x ; **polyharmonique** x³ ou x.

### Du « exact » au « lissé » : le réseau RBF (Broomhead & Lowe 1988 ; Moody & Darken 1989)

RBF = **amélioration de l'interpolation exacte, un lissage** : le nombre M de fonctions est **≪ N** et leurs paramètres (prototype, variance) sont **estimés par apprentissage**.

**Architecture à 2 couches** : entrées → unités RBF (transformation **non linéaire** : calcul de la distance entrée–prototype) → sorties **linéaires** (combinaison linéaire).

### Entraînement en 3 étapes (diapo 11)

1. **Placer les centres** (non supervisé) : k-means sur les entrées (ou sous-ensemble aléatoire d'exemples), M ≪ N.
2. **Choisir les largeurs** (heuristique) : σ = d\_max / √(2M) pour toutes les unités (d\_max = distance maximale entre centres), ou σ\_j = distance aux centres les plus proches.
3. **Ajuster les poids de sortie** (supervisé, **linéaire**) : moindres carrés **en une seule fois**, sans pas d'apprentissage : **Wᵀ = Φ⁺ D**.

(4ᵉ étape facultative : affiner centres, largeurs et poids ensemble par descente de gradient.) **Pourquoi c'est rapide** : la partie difficile et non linéaire n'a pas besoin d'étiquettes ; la partie supervisée est un problème linéaire.

**Dérivation de l'étape 3.** Erreur : R̃ = ½ Σ\_n Σ\_k ( y\_k(x(n)) − d\_k(n) )² = ½ || Φ Wᵀ − D ||², avec (Φ)*{nj} = φ\_j(x(n)) et (D)*{nk} = d\_k(n). Annuler le gradient par rapport à Wᵀ donne les **équations normales** :

```
Φᵀ Φ Wᵀ = Φᵀ D     ⟹     Wᵀ = Φ⁺ D,     Φ⁺ = ( ΦᵀΦ )⁻¹ Φᵀ      (pseudo-inverse)
```

**Dérivation de l'étape 4 (gradient sur la 1ʳᵉ couche).** Comme ∂φ\_j/∂σ\_j = φ\_j · ||x−μ\_j||²/σ\_j³ et ∂φ\_j/∂μ\_ji = φ\_j · (x\_i − μ\_ji)/σ\_j² :

```
∂R̃/∂σ_j  = Σ_n Σ_k ( y_k − d_k ) w_kj · exp(−||x−μ_j||²/2σ_j²) · ||x−μ_j||² / σ_j³
∂R̃/∂μ_ji = Σ_n Σ_k ( y_k − d_k ) w_kj · exp(−||x−μ_j||²/2σ_j²) · ( x_i − μ_ji ) / σ_j²
```

### Rôle de la largeur σ (diapo 12)

- **σ trop petit** (0,01) : pics isolés, la sortie retombe à 0 entre les centres → **sur-apprentissage** (MSE train 0,080 ; test 0,179).
- **σ bien choisi** (0,20) : les bosses voisines se **recouvrent un peu** → courbe lisse (train 0,015 ; test 0,027).
- **σ trop grand** (1,00) : chaque bosse couvre tout → **sous-apprentissage** (train 0,201 ; test 0,197).

C'est exactement le dilemme biais-variance (Partie III) appliqué à un hyperparamètre.

### Exercice XOR avec deux unités gaussiennes (diapo 15)

Centres (1,1) et (0,0) ; φ₁ = exp(−||x−(1,1)||²), φ₂ = exp(−||x−(0,0)||²).

| (x₁,x₂) | classe | φ₁ | φ₂ |
| --- | --- | --- | --- |
| (0,0) | 0 | e⁻² = 0,135 | e⁰ = 1,000 |
| (0,1) | 1 | e⁻¹ = 0,368 | e⁻¹ = 0,368 |
| (1,0) | 1 | 0,368 | 0,368 |
| (1,1) | 0 | 1,000 | 0,135 |

1. **Vérification** : pour (0,0), distance à (1,1) = √2 donc φ₁ = e^(−2) = 0,135 ; distance à (0,0) = 0 donc φ₂ = 1.
2. **(0,1) et (1,0) tombent au même point** car ils sont **à la même distance (1) des deux centres** (symétrie), donc φ₁ = φ₂ = e⁻¹.
3. **Poids** (ma solution, la diapo renvoie aux notes) : y = −φ₁ − φ₂ + 0,94 (w₁ = w₂ = −1, b ≈ 0,94). Classe 0 : φ₁+φ₂ = 1,135 → y = −0,20 < 0 ; classe 1 : 0,736 → y = +0,20 > 0. La droite de séparation est φ₁ + φ₂ = 0,935.

**Théorème de Cover** : une transformation non linéaire vers un nouvel espace rend (souvent) les classes **linéairement séparables**. Le XOR, impossible pour un perceptron, devient trivial dans l'espace (φ₁,φ₂).

### RBF pour la classification : lecture probabiliste (diapos 17–19)

Avec M fonctions radiales (composantes j), probabilité conditionnelle d'une classe : p(x|C\_k) = Σ\_j p(x|j) P(j|C\_k). Densité non conditionnelle : p(x) = Σ\_k p(x|C\_k)P(C\_k) = Σ\_j p(x|j)P(j), avec P(j) = Σ\_k P(j|C\_k)P(C\_k). Par Bayes :

```
P(C_k|x) = Σ_j [ P(j|C_k) p(x|j) P(C_k) / Σ_j' p(x|j')P(j') ] · P(j)/P(j) = Σ_j w_kj · φ_j(x)

1ʳᵉ couche :  φ_j(x) = p(x|j)P(j) / Σ_j' p(x|j')P(j') = P(j|x)         (« responsabilité » de la composante j)
2ᵉ couche :  w_kj  = P(j|C_k)P(C_k) / P(j) = P(C_k|j)
```

Interprétation : P(C\_k|x) = Σ\_j P(C\_k|j)·P(j|x) (loi des probabilités totales). Les unités cachées calculent **à quelle composante appartient x**, les poids de sortie donnent **la classe de chaque composante**.

**Théorème de Cotter (1990)** (Stone-Weierstrass) : sur un compact S de ℝⁿ, toute fonction continue t(x) peut être approchée à ε près par ψ(x,w) = Σ\_k w\_k φ(||x − c^k||) avec K centres : **comme le MLP, le RBF est un approximateur universel**.

### RBF vs MLP

|  | MLP | RBF |
| --- | --- | --- |
| Architecture | multicouche, connexions complexes | 2 couches, connexions spécialisées |
| Unités cachées | **codent** l'information (somme pondérée, hyperplans) | calculent des **distances** à des prototypes (bosses) |
| Représentation | **distribuée** | **localisée** |
| Apprentissage | global (rétropropagation) | en 2 phases (non supervisé + supervisé), plus rapide, moins d'exemples |

Frontières : un MLP trace des hyperplans ; un RBF entoure les classes par des régions localisées (ellipses). **Aujourd'hui** : noyau RBF des **SVM**, **processus gaussiens**, interpolation de données éparses (terrain, météo), modèles locaux rapides (contrôle, séries temporelles). **Limite** : en grande dimension, il faut beaucoup de bosses pour couvrir l'espace (**fléau de la dimension**).

---

## IV.3 LVQ : un dictionnaire de prototypes étiquetés (diapos 23–46)

**LVQ** (*Learning Vector Quantization*, Kohonen 1988) : adaptation de **k-means / quantification vectorielle** au cas **supervisé**.

- **Apprentissage** : déterminer un ensemble de **vecteurs de référence (prototypes)**, chacun associé à une classe ; l'espace est partitionné en régions (cellules de Voronoï).
- **Reconnaissance** : on affecte la **classe du prototype le plus proche** (règle du plus proche voisin, pour une distance donnée).
- **Vue réseau** : 2 couches complètement connectées : entrée (exemples) → sortie (classes), les prototypes de la classe C\_i étant regroupés.

**Résultat marquant (diapo 26)** : données de type XOR (chaque classe = 2 blobs), précision sur 1000 points de test : **1 prototype par classe = 50 %** (les moyennes de classe tombent au milieu : hasard) ; **2 prototypes par classe (LVQ1) = 98,7 %** (4 vecteurs stockés) ; **1-plus-proche-voisin avec les 400 points = 97,8 %**. → 4 prototypes bien placés classent aussi bien que stocker les 400 exemples.

**Frontières** : la frontière entre 2 classes est un sous-ensemble de la **tessellation de Voronoï** des prototypes (le dual est le réseau de **Delaunay**).

### LVQ1 (Kohonen)

Soit w(i) le prototype **le plus proche** de x (de classe C\_i), x ∈ C\_j.

```
Coût :  R_LVQ1(x,w) = + ||x − w(i)||²    si C_i = C_j     (bon prototype : on veut réduire la distance)
                      − ||x − w(i)||²    si C_i ≠ C_j     (mauvais prototype : on veut augmenter la distance)

Règles (gradient, car ∂||x−w||²/∂w = −2(x−w)) :
w(i)(t+1) = w(i)(t) + ε(t) ( x − w(i)(t) )    si C_i = C_j   → ATTIRER vers x
w(i)(t+1) = w(i)(t) − ε(t) ( x − w(i)(t) )    si C_i ≠ C_j   → REPOUSSER de x
w(k)(t+1) = w(k)(t)                              pour les autres prototypes (inchangés)
```

**Exemple chiffré (diapo 28)** : w\_A = (0,0) classe A, w\_B = (2,2) classe B, α = 0,5 ; x = (1,5 ; 1) de classe A.

1. Distances : ||x−w\_A|| = √(2,25+1) = 1,80 ; ||x−w\_B|| = √(0,25+1) = 1,12 → le plus proche est w\_B.
2. Classe de w\_B = B ≠ A → **mauvais** → repousser.
3. w\_B ← (2,2) − 0,5·(−0,5 ; −1) = **(2,25 ; 2,5)**.

*À toi de jouer (ma correction)* : on représente x de nouveau : ||x−w\_B|| = √(0,75²+1,5²) = 1,68 < 1,80, donc **w\_B est encore le plus proche** (mal classé : il faudra le repousser encore). Puis x′ = (0,5 ; 0,2) de classe A : ||x′−w\_A|| = 0,54 et ||x′−w\_B|| = 2,89 → w\_A le plus proche, bonne classe → **attirer** : w\_A ← (0,0) + 0,5·(0,5 ; 0,2) = **(0,25 ; 0,10)**.

### LVQ2 / LVQ2.1

On utilise les **deux** prototypes les plus proches : w(i) le **1ᵉʳ plus proche** (classe C\_i ≠ C\_k, donc **mauvaise** classe) et w(j) le **2ᵉ plus proche** (classe C\_j = C\_k, **bonne** classe), pour x ∈ C\_k.

```
Coût :  R_LVQ2(x,w) = ||x − w(j)||² − ||x − w(i)||²     si C_i ≠ C_k  et  x ∈ F_win ;   0 sinon
Fenêtre :  x ∈ F_win  ⟺  min( d(x,w(j))/d(x,w(i)) , d(x,w(i))/d(x,w(j)) ) > (1 − win)/(1 + win),    win ≈ 0,25
Règles :  w(i)(t+1) = w(i)(t) − ε(t)( x − w(i)(t) )      (on repousse le mauvais)
          w(j)(t+1) = w(j)(t) + ε(t)( x − w(j)(t) )      (on attire le bon)
```

Idée : on ne corrige que si x est **proche de la frontière** entre les deux prototypes (dans la fenêtre : les deux distances sont comparables), ce qui **déplace la frontière vers sa position optimale** (diapo 35). **LVQ3** : comme LVQ2.1, plus une petite mise à jour des deux prototypes s'ils sont de la même classe que x (pas ε·α), pour qu'ils restent sur les données. Défaut de LVQ2.1 : si on entraîne trop longtemps, les prototypes **dérivent**.

### GLVQ (Sato & Yamada, 1996) : LVQ avec fonction de coût

Pour chaque exemple x : d⁺ = distance au plus proche prototype de la **bonne** classe ; d⁻ = distance au plus proche prototype d'une **mauvaise** classe.

```
μ(x) = ( d⁺ − d⁻ ) / ( d⁺ + d⁻ )  ∈ [−1, 1]      μ < 0 ⟺ bien classé
E = Σ_x Φ( μ(x) ),     Φ = sigmoïde
```

Descente de gradient sur E : w⁺ **attiré** vers x, w⁻ **repoussé**. Avantages : **stable** (pas de dérive, contrairement à LVQ2.1 trop longtemps), maximise une **marge** (esprit SVM), prototypes restent lisibles. Les dérivées de μ (à vérifier) : ∂μ/∂d⁺ = 2d⁻/(d⁺+d⁻)² et ∂μ/∂d⁻ = −2d⁺/(d⁺+d⁻)² ; les poids de mise à jour de w⁺ et w⁻ sont donc proportionnels à d⁻ et d⁺ respectivement, multipliés par Φ'(μ) (grand pour les exemples proches de la frontière).

### ω-LVQ (Bennani, 1999) : apprendre les prototypes ET l'importance des variables

But simultané : (1) **pondérer les variables** selon leur pouvoir discriminant ; (2) classer comme LVQ2. On introduit une couche de poids π\_k (diagonale Π) : z = Πx.

```
Coût :  R_ω-LVQ2(x,w,Π) = ||Πx − w(j)||² − ||Πx − w(i)||²     si C_i ≠ C_k ;   0 sinon
Règles :  w(i)(t+1) = w(i)(t) − ε(t)( Πx − w(i)(t) )
          w(j)(t+1) = w(j)(t) + ε(t)( Πx − w(j)(t) )
          π_k(t+1) = π_k(t) + δ(t) x_k ( w_k(i)(t) − w_k(j)(t) )          (règle écrite sur la diapo)
Initialisation : Π(0) = I (poids π_ij = 1 si i = j, 0 sinon)
```

*Attention, à vérifier avec ton cours* : en dérivant le coût, ∂R/∂π\_k = 2x\_k ( w\_k(i) − w\_k(j) ), donc une descente de gradient donne π\_k ← π\_k **−** 2δ x\_k ( w\_k(i) − w\_k(j) ) = π\_k + 2δ x\_k ( w\_k(j) − w\_k(i) ). Intuition : la variable k est **renforcée** quand le bon prototype w(j) est plus proche de x que le mauvais w(i) sur cette variable. Le signe de la diapo semble dépendre de la convention des indices.

**Résultat (identification du locuteur, 3 mots de code par locuteur, Bennani, Fogelman, Gallinari 1990)** : LVQ2 de Kohonen = **96,6 %** (IC 95 % \[95,1 ; 98,1\]) ; ω-LVQ2 = **99,6 %** (IC 95 % \[99,2 ; 100\]).

### La famille LVQ

| Variante | Ce qui est mis à jour | Idée clé |
| --- | --- | --- |
| **LVQ1** | prototype le plus proche | attirer si bonne classe, repousser sinon |
| **LVQ2.1** | gagnant et second, de classes différentes | seulement dans une **fenêtre** autour de la frontière (win ≈ 0,2–0,3) |
| **LVQ3** | comme LVQ2.1 + les deux si même classe que x | petit pas supplémentaire ε·α : les prototypes restent sur les données |
| **GLVQ** | plus proche bon et plus proche mauvais | gradient d'un coût : stable, marge |
| **ω-LVQ** (Bennani) | prototypes + poids des variables | apprend l'importance de chaque variable |
| **GRLVQ / GMLVQ** (2002, 2009) | prototypes + pertinences / matrice de pertinence | apprend la **métrique** : pertinence interprétable |

En pratique : commencer par LVQ1 ou GLVQ ; ajouter l'apprentissage de pertinence pour savoir **quelles variables comptent**. **Pourquoi LVQ est encore utilisé (IA interprétable)** : prototypes lisibles (cas typiques inspectables par un expert), pertinence des variables, médecine/biologie (le médecin doit comprendre le modèle), parole et signaux. Outil : `sklearn-lvq` (GLVQ, GRLVQ, GMLVQ avec interface scikit-learn).

### Bilan de la Partie A

|  | MLP (poids) | RBF | LVQ |
| --- | --- | --- | --- |
| Connaissance stockée | poids | dictionnaire de centres | dictionnaire de prototypes étiquetés |
| Unités cachées calculent | sommes pondérées (hyperplans) | distances aux centres (bosses) | distances aux prototypes |
| Entraînement | rétropropagation globale | k-means puis moindres carrés | attirer / repousser |
| Interprétabilité | faible | moyenne | **élevée** |

Fil conducteur : **un petit dictionnaire de référents bien placés, et des décisions par similarité**. La suite (partie B) construit le dictionnaire **sans aucune étiquette**.

---

## IV.4 Le principe de l'apprentissage non supervisé (diapos 47–52)

- **Supervisé** : chaque x vient avec une étiquette y ; but : apprendre f : x ↦ y (un professeur corrige chaque réponse). **Non supervisé** : on n'a que x ; but : **découvrir la structure de p(x)** (groupes, directions principales, cartes). Pourquoi : les étiquettes coûtent cher, les données brutes abondent.
- **Principe** : système entrée-sortie **sans professeur**, qui découvre spontanément dans les données des **caractéristiques, régularités, corrélations, catégories**. Les unités et connexions doivent être un **effet d'organisation (auto-organisation)**.
- **Condition (Barlow 1989)** : l'apprentissage non supervisé est efficace s'il y a de la **redondance** dans les données ; sinon les données ≃ bruit. **La redondance produit de la connaissance.**
- **Ce que peuvent représenter les sorties** : une **similarité** (entre une forme nouvelle et des formes passées), des **composantes principales** (axes de projection), un **clustering** (groupes homogènes), des **prototypes** (représentant de chaque groupe), un **codage** (réduction de dimension, quantification vectorielle), une **projection** qui préserve la topologie.

**Deux grandes familles :**

|  | **Apprentissage hebbien** | **Apprentissage compétitif** |
| --- | --- | --- |
| Outil | règle de Hebb | distance |
| Sorties actives | **plusieurs** (valeurs continues : similarités ou projections) | **une seule** (ou une par groupe) : *Winner-Take-All*, le gagnant emporte tout |
| Méthodes | Hebb (1949), Oja (1982) → ACP en ligne, Sanger | quantification vectorielle, k-means ; SOM (+ voisinage sur grille) ; Neural Gas (+ voisinage par rang) ; GNG (+ nouvelles unités) ; ART (+ test de vigilance) |

Troisième branche : **non supervisé profond** (auto-encodeurs, modèles génératifs, auto-supervisé).

---

## IV.5 Apprentissage hebbien et ACP (diapos 53–58)

### Règle de Hebb (1949)

Unité linéaire : y = Σ\_i w\_i x\_i. Avec une seule sortie :

```
Δw_i = α · y · x_i          (α = pas d'apprentissage)
```

« Les neurones qui s'activent ensemble se connectent » : les données fréquentes influencent davantage l'apprentissage. **Problème : croissance incontrôlée des poids.** Raison : Δ(||w||²) ≈ 2αy² ≥ 0 à chaque pas, donc la norme ne peut que croître ; plus précisément, la composante de w sur le vecteur propre principal de la matrice de corrélation est multipliée par (1 + αλ) à chaque pas : croissance **exponentielle**. Simulation du cours : ||w|| ≈ **4 × 10⁶** après 1000 pas avec Hebb.

### Règle d'Oja (1982) : Hebb + contrainte de norme

On impose ||w|| = 1 (Σ w\_i² = 1) :

```
Δw_i = α · y · ( x_i − y · w_i )          (x_i = entrée ; y w_i = « sortie rétropropagée » qui retranche)
```

**À l'équilibre** (⟨Δw\_i⟩ = 0), avec R = ⟨xxᵀ⟩ la **matrice de corrélation** et y = wᵀx :

```
0 = ⟨y x_i − y² w_i⟩ = Σ_j R_ij w_j − ( wᵀ R w ) w_i      ⟺      R w = ( wᵀ R w ) w = λ w
```

donc **w est un vecteur propre de R**, de valeur propre λ = wᵀRw, et ||w|| = 1. La règle choisit la direction qui **maximise ⟨y²⟩ = wᵀRw** : le **1ᵉʳ axe principal**. Simulation : ||w|| se stabilise à **1,00**, angle de **0,6°** avec la 1ʳᵉ composante principale. → **Règle d'Oja = ACP en ligne avec un seul neurone linéaire.**

### Règle de Sanger (GHA, 1989) : plusieurs composantes

m neurones linéaires, y\_i = w\_i · x :

```
Δw_ij = α · y_i · ( x_j − Σ_{k ≤ i} y_k w_kj )
```

Le neurone 1 suit la règle d'Oja → PC1 ; le neurone 2 voit l'entrée **moins ce que le neurone 1 explique déjà** → PC2 ; etc. (**déflation**). Les lignes de W convergent vers les m premiers vecteurs propres, dans l'ordre. Le réseau calcule l'ACP **exemple par exemple**, sans jamais former la matrice de covariance.

### Mémoire hebbienne : le réseau de Hopfield (1982)

N neurones binaires s\_i = ±1, complètement connectés. **Stockage** de P motifs ξᵖ par la règle de Hebb :

```
W = (1/N) Σ_p ξᵖ ξᵖᵀ,     w_ii = 0
Rappel : répéter  s_i ← signe( Σ_j w_ij s_j )  jusqu'à stabilité
```

Une **énergie** (à vérifier : E = −½ sᵀWs) **décroît** à chaque pas : l'état tombe dans l'**attracteur** le plus proche (un motif stocké). **Capacité ≈ 0,14 N** motifs. C'est une **mémoire adressable par le contenu** : un indice partiel ou bruité (25 % de pixels inversés) retrouve la mémoire complète (3 lettres H, T, O dans 100 neurones). Hopfield (prix Nobel de physique 2024 avec Hinton) ; les réseaux de Hopfield modernes sont proches de l'**attention** (Partie V).

---

## IV.6 Apprentissage compétitif et k-means (diapos 59–70)

### Notations

- M **unités** A = {c₁,…,c\_M}, chacune avec un **prototype** (vecteur de référence) w\_k ∈ ℝ^D.
- Un ensemble C ⊂ A×A de **connexions de voisinage**, **symétriques** et sans valeur : (i,j) ∈ C ⇔ (j,i) ∈ C ; **N\_k = { i ∈ A | (k,i) ∈ C }** = voisins de k.
- Données générées selon p(x), jeu fini D\_N = {x(1),…,x(N)}.
- **Unité gagnante** : s(x) = argmin\_{c∈A} ||x − w\_c||.
- **Région de Voronoï** de i : V\_i = { x ∈ ℝ^D | i = argmin\_j ||x − w\_j|| } (tessellation de Voronoï ou de Dirichlet) ; cas fini : R\_c = { x ∈ D\_N | s(x) = c }. Son dual est la **triangulation de Delaunay**.
- **Principe** : avant l'apprentissage les prototypes sont n'importe où ; après, chaque prototype est au centre d'un groupe (*cluster*) d'exemples.

### Critères

```
Erreur de quantification (distorsion) :  E(p,A) = Σ_c ∫_{V_c} ||x − w_c||² p(x) dx      cas fini : E(D,A) = Σ_c Σ_{x∈R_c} ||x − w_c||²
Maximisation de l'entropie :            H(X) = − Σ P(x) log P(x)        (toutes les unités servent à peu près autant)
Projection topologique :                 les voisins sur la carte représentent des données voisines
```

### Cas simple : seul le gagnant s'adapte

```
sortie y_{i*} = 1, y_i = 0 pour i ≠ i* :   Δw_{i*j} = α ( x_j − w_{i*j} )     ⇔   Δw_ij = α y_i ( x_j − y_i w_ij )  (règle d'Oja)
```

### Apprentissage compétitif = k-means en ligne

Pour chaque exemple x : (1) trouver le gagnant s = argmin\_i ||x − w\_i|| ; (2) le rapprocher : **w\_s ← w\_s + ε ( x − w\_s )** ; (3) les autres prototypes ne bougent pas.

**Pourquoi « k-means » ?** Avec ε = 1/n\_s (n\_s = nombre d'exemples déjà gagnés par s), w\_s est exactement la **moyenne** de ses exemples. Dérivation : w^(n) = w^(n−1) + (1/n)( x\_n − w^(n−1) ) ⟺ w^(n) = ((n−1)/n) w^(n−1) + x\_n/n, qui est la moyenne courante. **Version par lots** : algorithme de **Lloyd / LBG** : affecter tous les points à leur centre le plus proche, puis recalculer chaque moyenne, et répéter.

### Exercice : k-means à la main (diapo 68)

Données 1-D : x = 1, 2, 6, 8, 9 ; K = 2 ; centres initiaux c₁ = 1, c₂ = 2.

- **Itération 1** : 1 → c₁ (distances 0 / 1) ; 2 → c₂ (1 / 0) ; 6 → c₂ (5 / 4) ; 8, 9 → c₂. Groupes {1} et {2,6,8,9}. Nouveaux centres : c₁ = 1 ; c₂ = 25/4 = **6,25**.
- **Itération 2** : 1 → c₁ ; 2 → c₁ (distances 1 / 4,25) ; 6, 8, 9 → c₂. Groupes {1,2} et {6,8,9}. Centres : c₁ = **1,5** ; c₂ = 23/3 ≈ **7,67**.
- **Itération 3** : aucune affectation ne change → **convergence** (2 mises à jour de centres, la 3ᵉ passe confirme la stabilité).
- **Inertie finale** Σ(x−c)² = (0,5² + 0,5²) + (1,67² + 0,33² + 1,33²) = 0,5 + 4,67 ≈ **5,17**.
- **Pourquoi le 1ᵉʳ découpage est mauvais** : les deux centres initiaux sont **collés** à gauche, donc un centre ne capte qu'un point et l'autre « avale » presque tout. **Meilleure initialisation** : k-means++ (centres initiaux éloignés les uns des autres), ou prendre des exemples tirés au hasard et relancer plusieurs fois (plusieurs graines).

### Application : compression d'image (diapo 69)

Chaque pixel est un point de l'espace RGB (3-D) ; k-means trouve K couleurs prototypes (la **palette**, ou *codebook*) ; chaque pixel est stocké comme l'**indice** de son prototype. Original : 24 bits/pixel ; **K = 16** : log₂(16) = 4 bits/pixel, **6 fois plus petit** ; **K = 4** : 2 bits/pixel, **12 fois plus petit** (mais les collines vertes disparaissent : trop peu de prototypes). Même principe : palettes GIF, *codebooks* de codecs de parole, compression de poids de réseaux.

### Limites du « winner-take-all »

1. **Unités mortes** : un prototype initialisé loin des données ne gagne jamais, donc n'apprend jamais.
2. **Initialisation** : le résultat dépend des positions de départ (minima locaux).
3. **Pas de topologie** : des unités voisines ne représentent pas forcément des données voisines (rien à visualiser).
4. **K fixe** : le nombre de prototypes doit être choisi à l'avance.

Remèdes : faire apprendre aussi les voisins (**SOM** voisinage sur grille ; **Neural Gas** voisinage par rang → corrige 1, 2, 3) ; faire **grandir** le réseau (**GNG**, **ART** → corrige 4).

---

## IV.7 Cartes auto-organisatrices de Kohonen (SOM) (diapos 71–91)

**SOM** (*Self-Organizing Map*, Kohonen 1982, d'après Willshaw & von der Malsburg 1976) : **projeter des données de grande dimension sur un espace discret, régulier, de dimension plus petite** (1-D ou 2-D). Les unités occupent les nœuds d'un **réseau régulier** (grille), avec une notion de **voisinage** entre unités. **Propriété** : **des données semblables ont des projections proches sur la carte.**

### Voisinage et décroissance

Distance sur la grille entre les unités r = c\_{km} et s = c\_{ij} (distance de Manhattan) : **d₁(r,s) = |i−k| + |j−m|**. Grille **hexagonale** ou **rectangulaire**. Fonction de voisinage (gaussienne) :

```
h_rs(t) = ( 1/λ(t) ) · exp( − d₁²(r,s) / λ²(t) ),         λ(t) = λ_i ( λ_f / λ_i )^{ t / t_max }
```

Le rayon λ(t) **diminue** (décroissance exponentielle de λ\_i à λ\_f) : au début, grand voisinage = ordre global ; à la fin, voisinage étroit = ajustement fin.

### Fonction de coût

```
BMU(x(n)) = argmin_s ||w_s − x(n)||²                     (Best Matching Unit = unité gagnante)
R̃(w) = (1/N) Σ_n Σ_j h_{j,BMU(x(n))} ||w_j − x(n)||²            ∇_{w_j} R̃(n) = h_{j,BMU(x(n))} ( w_j − x(n) )
```

C'est la distorsion de k-means **pondérée par le voisinage**.

### Algorithme d'apprentissage (diapo 80)

1. **Initialisation** : A = grille M₁×M₂, prototypes w\_c ∈ ℝⁿ aléatoires, voisinage C (hexagonal/rectangulaire), t = 0.
2. **Présenter** un exemple x.
3. **Gagnant** : s = argmin\_{c∈A} ||x − w\_c||² (phase compétitive).
4. **Adapter** (phase d'adaptation) : **w\_r(t+1) = w\_r(t) + ε(t) · h\_rs(t) · ( x − w\_r(t) )** pour tout r, avec ε(t) = ε\_i ( ε\_f / ε\_i )^{ t/t\_max }.
5. t ← t + 1.
6. **Test d'arrêt** : si t < t\_max, retourner en 2.

Paramètres du cours : λ\_i = 2, λ\_f = 0,5 ; ε\_i = 0,5, ε\_f = 0,005 ; t\_max = 10 000.

### Exemple chiffré : une mise à jour de SOM (diapo 81)

Carte 1-D de 3 unités w₁ = (0,0), w₂ = (1,0), w₃ = (2,0) ; x = (1,8 ; 1) ; ε = 0,5 ; voisinage h = 1 (gagnant), 0,5 (voisin direct), 0 (autres).

| Unité | distance de x à w | h | Mise à jour ε·h·(x−w) | Nouveau prototype |
| --- | --- | --- | --- | --- |
| w₁ | 2,06 | 0 | — | (0, 0) |
| w₂ | 1,28 | 0,5 | 0,25 × (0,8 ; 1) = (0,2 ; 0,25) | **(1,2 ; 0,25)** |
| w₃ (gagnant) | 1,02 | 1 | 0,5 × (−0,2 ; 1) = (−0,1 ; 0,5) | **(1,9 ; 0,5)** |

k-means ne déplacerait que w₃. La SOM entraîne aussi w₂, voisin du gagnant : c'est ce qui **garde la carte ordonnée**. **Réponse au quiz Q2** : si le voisinage se réduit au seul gagnant (h = 1 pour le gagnant, 0 sinon), on retrouve **k-means en ligne**.

### Taille de la carte (heuristique, diapo 82)

On estime le nombre total de neurones **M ≈ 5 · N^0,54321** (N = nombre d'exemples), puis on fait une **ACP** et on récupère les deux plus grandes valeurs propres V₁ et V₂ ; on pose R = √(V₁/V₂) et on cherche n₁, n₂ tels que **n₁ · n₂ = M** et **n₁/n₂ = R**. En résolvant : **n₁ = √(M·R)** et **n₂ = √(M/R)** (à arrondir). La grille est donc d'autant plus allongée que les données le sont.

### Phase d'auto-organisation

Le réseau part d'un nuage désordonné (grille tordue au hasard) puis **se déplie** progressivement jusqu'à épouser la forme des données (diapos 83–88 : grille qui se déplie, anneau, image, IRIS).

### Utiliser une SOM en pratique (diapo 90)

- **Normaliser** les variables d'abord ; **initialiser** sur le plan des deux premières composantes principales ; entraîner en **deux phases** : *ordonnancement* (voisinage large, ε grand) puis *convergence* (voisinage étroit, ε petit) ; préférer la grille **hexagonale**, dimensionnée par l'heuristique ACP.
- **Qualité** :
  - **Erreur de quantification** QE = (1/N) Σ\_n ||x\_n − w\_{s(x\_n)}|| : **fidélité** (les prototypes représentent-ils bien les données ?).
  - **Erreur topographique** TE = part des x\_n dont les **1ᵉʳ et 2ᵉ meilleures unités ne sont pas voisines** sur la grille : **organisation** (la carte est-elle dépliée ?).
  - Les deux erreurs évoluent souvent **en sens opposé** : chercher un compromis.
- **Lecture** : **U-matrix** (distances entre prototypes voisins : les crêtes séparent les groupes), **plans de composantes** (une carte par variable). Outils : MiniSom (Python), `kohonen` (R), SOM Toolbox (MATLAB).
- **Usages** : analyse exploratoire (carte de la pauvreté mondiale, 39 indicateurs, Kaski & Kohonen 1996), cartes de documents (WEBSOM), surveillance de procédés (une dérive = une trajectoire sur la carte), segmentation (clients, risque de crédit, expression de gènes : une cellule = un profil type). Fil commun : une **carte 2-D lisible** par un expert.

---

## IV.8 Neural Gas (NG) et Growing Neural Gas (GNG) (diapos 92–108)

Question : et si on **supprime la grille** et qu'on laisse le réseau découvrir **sa propre topologie** ?

### Neural Gas (Martinetz & Schulten, 1991)

Les prototypes sont adaptés **indépendamment de tout arrangement topologique prédéfini**. L'information d'arrangement est donnée implicitement par les **distorsions** associées à chaque x : D\_x = { ||x − w\_i|| , i = 1..M }. L'adaptation dépend de l'**ordre (rang)** des éléments de D\_x :

```
Δw_i = α · f_i(D_x) · ( x − w_i ),      f_i(D_x) ∈ [0,1]      →  règle « Winner-Take-Most » (le gagnant prend le plus) au lieu de « Winner-Take-All »
```

Pour chaque unité i, on note **k\_i** son **rang** : le nombre d'unités j **plus proches de x** que i (||x − w\_j|| < ||x − w\_i||). Le gagnant a k = 0.

```
f(k_i) = exp( − k_i / λ ),      f(0) = 1,   f(k) → 0 quand k → ∞      (λ = nombre d'unités qui adaptent significativement leur prototype)
w_i(t+1) = w_i(t) + ε(t) · exp( − k_i / λ ) · ( x − w_i(t) )
```

Le voisinage est donc défini par **le rang dans l'ordre des distances**, et non par la position sur une grille. Toutes les unités apprennent (donc pas d'unité morte).

**Règle hebbienne compétitive** : pour **mémoriser la topologie**, on crée une **connexion** entre l'unité gagnante i₀ et la **2ᵉ plus proche** i₁ : C\_{i₀i₁} = 1, avec un **âge** t\_{i₀i₁} qui doit rester < T (durée de vie).

**Algorithme** : (0) w\_i aléatoires, C = ∅ ; (1) présenter x ; (2) classer les unités par distance croissante (i₀, i₁, …, i\_{M−1}) ; (3) adapter tous les poids avec exp(−k\_i/λ) ; (4) si C\_{i₀i₁} = 0, créer la connexion (C ← 1, âge ← 0), sinon remettre son âge à 0 ; (5) **incrémenter l'âge** de toutes les connexions issues de i₀ ; (6) **supprimer** les connexions d'âge > T ; retour en (1).

Résultat : un **graphe** qui épouse la forme des données (anneau, formes en L ou en Y…), y compris les **trous**.

### Growing Neural Gas (Fritzke, 1994/1995)

**Version incrémentale** : le réseau **crée à la fois des unités et des connexions topologiques** ; il **grandit** au lieu d'avoir un M fixé. Pour créer des unités : on mémorise l'**erreur locale** accumulée par chaque unité, et on crée une nouvelle unité près de celle qui a accumulé la **plus grande erreur**.

1. **Initialisation** : A = {c₁, c₂}, C = ∅.
2. **Présenter** x.
3. Déterminer le **1ᵉʳ gagnant** s₀ = argmin\_c ||x−w\_c|| et le **2ᵉ** s₁ = argmin\_{c≠s₀} ||x−w\_c||.
4. Si la connexion (s₀,s₁) n'existe pas, la **créer** (âge 0).
5. **Accumuler l'erreur** du gagnant : ΔE\_{s₀} = ||x − w\_{s₀}||².
6. **Adapter** : w\_{s₀} ← w\_{s₀} + ε\_b (x − w\_{s₀}) pour le gagnant, et w\_i ← w\_i + ε\_n (x − w\_i) pour tout voisin i ∈ N\_{s₀} (ε\_b > ε\_n).
7. **Vieillir** les connexions issues de s₀ : t\_{s₀ i} ← t\_{s₀ i} + 1.
8. **Supprimer** les connexions d'âge > T, puis les unités **sans connexion**.
9. **Créer une nouvelle unité** (en pratique, périodiquement, tous les λ pas) :
   - q = unité d'**erreur accumulée maximale** ; g = parmi les voisins de q, celle d'erreur maximale ;
   - nouvelle unité r : **w\_r = (w\_q + w\_g)/2** ;
   - connecter r à q et g, **supprimer** la connexion (q,g) ;
   - réduire l'erreur de q et g d'une fraction δ : ΔE\_q = −δE\_q, ΔE\_g = −δE\_g ;
   - erreur de r : E\_r = (E\_q + E\_g)/2.

**Réponse au quiz Q4** : GNG ajoute une unité **au milieu de la zone qui a accumulé le plus d'erreur** (entre l'unité la plus fautive q et son voisin le plus fautif g). Usages : apprentissage de topologie (trous, branches, nombre de composantes connexes), reconstruction 3-D de surfaces (nuage de points → maillage), cartes pour la robotique, flux de données (variantes avec oubli). **Force : le graphe de prototypes est lui-même le résultat**, un squelette compact des données. Démo interactive : demogng.de.

---

## IV.9 Adaptive Resonance Theory (ART) (diapos 109–117)

**Dilemme stabilité–plasticité** : **plasticité** = apprendre de nouvelles choses à partir de nouvelles données ; **stabilité** = garder ce qui est déjà appris. Les deux sont en tension. Toujours ouvert aujourd'hui : un réseau profond entraîné sur une nouvelle tâche peut **oublier la précédente** (*catastrophic forgetting*, sujet de l'apprentissage continu).

**ART (Grossberg, 1987)** : système qui acquiert en continu de nouvelles connaissances **sans oublier** les anciennes ; composé de modules obéissant à des équations différentielles souvent non linéaires ; **le nombre de catégories est dynamique**. **Réponse d'ART** : on ne met à jour un prototype que si la nouvelle entrée lui **ressemble assez** (**test de vigilance**) ; sinon on **crée un nouveau prototype**.

### ART1 (entrées binaires)

1. Présenter x. 2. **Gagnant** s = argmax\_j Σ\_i w\_ij x\_i.
2. **Test de vigilance** : la sortie passe le test ssi  Σ\_i t\_is x\_i / Σ\_i x\_i > δ  (δ = seuil de vigilance, t = poids de rétroaction binaires).
   - Si le gagnant échoue : le **masquer** et retourner en 2 ; répéter jusqu'à ce qu'un gagnant passe le test.
   - Si **aucune** sortie ne passe : **créer une nouvelle unité** et retourner en 2.
3. Si un gagnant passe : adapter  w\_is(t+1) = t\_is(t) x\_i / ( 0,5 + Σ\_i t\_is(t) x\_i )  et  t\_is(t+1) = t\_is(t) x\_i.

La mise à jour de t\_is = t\_is·x\_i fait l'**intersection** (ET logique) entre le prototype et l'entrée : le prototype ne garde que les traits communs.

### ART2 (entrées continues)

1. Présenter x. 2. Gagnant s = argmin\_j ||x − w\_j||. 3. **Test** : passe ssi ||x − w\_s|| < δ (**δ = rayon du cluster**).
2. Si le gagnant échoue : **nouvelle unité** k avec w\_k = x.
3. Sinon : w\_s(t+1) = ( x + w\_s(t) · |cluster\_s(t)| ) / ( |cluster\_s(t)| + 1 ), c'est-à-dire la **moyenne courante** du cluster (|A| = cardinal de A).

**Réponse au quiz Q5** : le paramètre de **vigilance** contrôle la **taille / le nombre des catégories** : vigilance élevée (rayon δ petit en ART2) → beaucoup de petites catégories ; vigilance faible → peu de grandes catégories.

---

## IV.10 Auto-encodeurs (rappel Partie III, ici côté non supervisé) (diapos 118–122)

Un auto-encodeur apprend un bon **code** en essayant de **copier son entrée** : x → encodeur → code z → décodeur → x̂, avec **L = ||x − x̂||²**.

- **Auto-encodeur linéaire = ACP** (même sous-espace, Baldi & Hornik 1989 ; ce que trouve la règle d'Oja). **Auto-encodeur non linéaire profond** : découvre des structures courbes (**variétés**) que l'ACP ne voit pas. Variantes : débruiteur, parcimonieux, variationnel.
- **Le piège** : un code **assez large** permet de **recopier** l'entrée : reconstruction parfaite, mais on n'apprend rien. D'où une **contrainte** qui force le code à garder l'essentiel :
  - **sous-complet** : code plus petit que l'entrée (goulot d'étranglement, il faut compresser) ;
  - **débruitant** (Vincent et al. 2008) : on **corrompt** l'entrée (x̃) et on demande la version **propre** x : le réseau doit apprendre la structure ;
  - **parcimonieux** : pénalité ||z||₁ : seulement quelques unités du code s'activent pour chaque entrée. **Réponse au quiz Q6** : un code très large n'impose aucune compression, donc l'identité est une solution triviale : le réseau recopie sans extraire de structure.

### Auto-encodeur variationnel (VAE, Kingma & Welling 2014)

x → encodeur → (μ, σ) → **z = μ + σ·ε** avec ε \~ N(0, I) (**astuce de reparamétrisation** : le tirage aléatoire est isolé dans ε, ce qui laisse passer le gradient) → décodeur → x̂.

```
Perte = || x − x̂ ||²   +   KL( q(z|x) || N(0, I) )
         (reconstruction)   (régularisation : garder les codes proches d'une gaussienne standard)
Pour q = N(μ, diag σ²) (à vérifier) :  KL = −½ Σ_i ( 1 + log σ_i² − μ_i² − σ_i² )
```

**Génération** : tirer z \~ N(0, I) et le passer dans le décodeur : nouvel échantillon réaliste ; des codes voisins donnent des sorties voisines (**espace latent lisse**). Base de nombreux modèles génératifs, dont l'espace latent de Stable Diffusion.

### Application : détection d'anomalies par reconstruction (diapo 122)

**Recette** : entraîner **uniquement sur des données normales** ; tout ce que le modèle reconstruit mal est suspect. Seuil = **99ᵉ percentile** des erreurs d'entraînement (1,82 dans l'exemple). Simulation (données 10-D proches d'un plan 2-D, auto-encodeur linéaire = ACP à 2 unités de code) : **92,5 % d'anomalies détectées** pour **0,8 % de fausses alarmes** sur les données normales. Usages : fraude, maintenance prédictive, intrusion réseau, défauts en production.

---

## IV.11 Visualisation et apprentissage auto-supervisé (diapos 123–127)

### Voir les données : ACP vs t-SNE / UMAP

Sur 1 797 chiffres manuscrits (8×8 = 64 dimensions), la projection **ACP** (linéaire) mélange les classes ; **t-SNE** (2008) et **UMAP** (2018) (non linéaires) font apparaître des groupes nets : ils gardent les **plus proches voisins** de chaque point proches, comme une SOM mais **continue et sans grille**. **Attention** : la **taille des groupes** et les **distances entre groupes** ne sont **pas interprétables**.

### Apprentissage auto-supervisé : les données s'étiquettent elles-mêmes

- **Cacher et prédire** : **GPT** prédit le mot suivant ; **BERT** remplit des mots masqués ; **MAE** reconstruit des morceaux d'image masqués. *L'étiquette est un morceau des données elles-mêmes.*
- **Contrastif (SimCLR, 2020)** : deux augmentations aléatoires (recadrage, couleur…) d'une même image doivent avoir des **embeddings proches** (on les rapproche), tandis que les autres images sont **repoussées**. (Perte usuelle, à vérifier : ℓ = −log \[ exp(sim(z\_i,z\_j)/τ) / Σ\_{k≠i} exp(sim(z\_i,z\_k)/τ) \].)
- **Ensuite** : *fine-tuning* avec quelques étiquettes. On apprend sans étiquettes d'abord, on adapte avec peu d'étiquettes, à l'échelle de milliards d'exemples.

### Clustering profond

(1) **Pré-entraîner** un auto-encodeur pour obtenir un embedding compact ; (2) **k-means** dans l'espace d'embedding (prototypes initiaux) ; (3) **affiner les deux** : durcir les affectations souples et mettre à jour l'encodeur pour que les groupes se resserrent. Exemples : **DEC** (Xie et al. 2016), SwAV, DINO, SOM profondes empilées (équipe du cours). **Héritiers de k-means et de la SOM : des prototypes, mais dans un espace appris.**

### Panorama de l'apprentissage non supervisé aujourd'hui

Auto-encodeurs et VAE (compresser, générer) ; **GAN** (générateur contre discriminateur) ; **modèles de diffusion** (apprendre à retirer le bruit pas à pas : les générateurs d'images actuels) ; **auto-supervisé** (tâche inventée à partir des données brutes) ; **clustering profond** ; **modèles de fondation** (les LLM sont pré-entraînés sur du texte brut, sans étiquettes ; Partie V). Fil commun : **extraire de la structure des données sans étiquettes**.

---

## IV.12 Choisir, évaluer et appliquer (diapos 128–132)

### Les méthodes compétitives en un coup d'œil

| Méthode | Nombre d'unités | Topologie | Qui apprend ? | Idéal pour |
| --- | --- | --- | --- | --- |
| **k-means** | fixe | aucune | le gagnant seul | clustering simple et rapide |
| **SOM** | fixe | **imposée** (grille 1-D / 2-D) | gagnant + voisins sur la grille | visualisation, projection |
| **Neural Gas** | fixe | **apprise** (arêtes) | toutes les unités, par rang | faible distorsion, robuste |
| **Growing NG** | croît / décroît | apprise (arêtes) | gagnant + voisins | forme et taille inconnues |
| **ART** | croît (vigilance) | aucune | le gagnant s'il est vigilant | apprentissage continu, en ligne |

De haut en bas : **moins de structure imposée, plus de structure apprise**.

### Sans étiquettes, comment savoir que ça a marché ?

- **Critères internes** : erreur de quantification (inertie), **silhouette** (à vérifier : s = (b − a)/max(a, b), a = distance moyenne intra-groupe, b = distance moyenne au groupe voisin le plus proche), erreur topographique.
- **Visualisation** : projections 2-D, U-matrix, plans de composantes.
- **Stabilité** : relancer avec d'autres graines ou sous-échantillons : le résultat est-il similaire ?
- **Vérification externe** : quelques étiquettes (**ARI, NMI**) ou utilité pour une tâche en aval.
- **Méthode du coude (*elbow*)** : tracer l'inertie de k-means en fonction de K ; la courbe se **coude** au bon K (K = 3 dans l'exemple : l'inertie chute de 2000 à 250 environ puis stagne).

### Applications

| Application | Méthodes typiques | Exemples |
| --- | --- | --- |
| Segmentation | k-means, SOM | profils clients, segments de marché, groupes de patients |
| Exploration et visualisation | SOM, ACP, t-SNE / UMAP | cartes socio-économiques, biologie unicellulaire |
| Compression | k-means / quantification vectorielle, auto-encodeurs | palettes de couleurs, *codebooks* de parole |
| Détection d'anomalies | auto-encodeurs, distance aux prototypes | fraude, maintenance prédictive, intrusion |
| Organisation de documents et d'images | SOM, embeddings + clustering | recherche, recommandation, dédoublonnage |
| Topologie et forme | Neural Gas, GNG | reconstruction 3-D, cartes de robots |
| Pré-entraînement | auto-supervisé | LLM, modèles de fondation en vision |

### Quelle méthode pour quel objectif ?

| Je veux… | Essayer | Conseil |
| --- | --- | --- |
| visualiser des données de grande dimension | SOM, ACP, t-SNE / UMAP | la SOM donne une carte stable avec prototypes |
| trouver des groupes, K connu | k-means, Neural Gas | lancer plusieurs initialisations |
| trouver des groupes, K inconnu | GNG, ART, k-means + coude / silhouette | vérifier la stabilité du résultat |
| réduire la dimension, compresser | ACP (Oja, GHA), auto-encodeur, VQ | essayer le linéaire d'abord, puis le non linéaire |
| détecter des anomalies | auto-encodeur, distance aux prototypes | entraîner sur données normales uniquement |
| apprendre sur un flux de données | k-means en ligne, ART, GNG | surveiller la dérive des données |
| préparer des caractéristiques pour une tâche ultérieure | auto-supervisé | *fine-tuning* avec peu d'étiquettes |

---

## IV.13 Quiz de la Partie IV avec corrigé

1. **Pourquoi la règle de Hebb diverge-t-elle, et comment Oja corrige-t-il ?** Hebb : Δw = α y x ne peut que faire croître ||w|| (la norme croît exponentiellement). Oja ajoute le terme **−α y² w** qui fait revenir la norme à 1 ; à l'équilibre w est le vecteur propre principal de la matrice de corrélation (ACP en ligne).
2. **Quel algorithme obtient-on quand le voisinage de la SOM se réduit au gagnant ?** **k-means en ligne** (apprentissage compétitif winner-take-all).
3. **Qu'est-ce qu'une unité morte, et quelles méthodes l'évitent ?** Un prototype initialisé loin des données qui ne gagne jamais, donc n'apprend jamais. Évitée par la **SOM** et le **Neural Gas** (les voisins / toutes les unités apprennent), le **GNG** et **ART** (nouvelles unités créées là où il faut), ou en initialisant sur des exemples.
4. **Comment GNG décide-t-il où ajouter une unité ?** Au milieu de l'unité d'**erreur accumulée maximale** q et de son voisin d'erreur maximale g : w\_r = (w\_q + w\_g)/2.
5. **Que contrôle la vigilance dans ART ?** Le **degré de ressemblance exigé** pour mettre à jour un prototype existant, donc la **taille et le nombre des catégories**.
6. **Pourquoi un auto-encodeur à code très large n'apprend-il rien d'utile ?** Il peut recopier l'entrée (identité) sans compresser ni extraire de structure ; il faut une contrainte (goulot, bruit, parcimonie).

---

## IV.14 Idées à retenir et checklist (Partie IV)

1. L'apprentissage non supervisé trouve de la **structure** dans des données non étiquetées et **a besoin de redondance**.
2. **Hebb** renforce les connexions co-actives : avec **Oja / Sanger** il calcule l'**ACP** ; dans **Hopfield** il stocke des mémoires.
3. L'apprentissage **compétitif** rapproche le gagnant de chaque exemple : **k-means en ligne**, quantification vectorielle.
4. La **SOM** met aussi à jour les voisins du gagnant sur la grille : carte qui **préserve la topologie**, idéale pour la visualisation.
5. **Neural Gas** apprend sa topologie ; **GNG** apprend sa taille ; **ART** apprend sans oublier.
6. Les **auto-encodeurs** apprennent des codes non linéaires par reconstruction : compression, anomalies, génération (VAE).
7. L'**auto-supervisé** généralise ces idées : les modèles de fondation apprennent sans étiquettes.
8. Sans étiquettes, il faut une **évaluation soignée** : critères internes, visualisation, stabilité.

À savoir refaire : (a) construire un RBF en 3 étapes et **dériver Wᵀ = Φ⁺D** ; (b) l'exercice **XOR avec 2 gaussiennes** ; (c) **une mise à jour LVQ1** et la différence LVQ1 / LVQ2.1 / GLVQ ; (d) **dériver l'équilibre d'Oja** (Rw = λw) ; (e) **k-means à la main** et sa moyenne courante ; (f) **une mise à jour de SOM** avec voisinage ; (g) l'algorithme **NG** (rang, exp(−k/λ), connexions et âges) et **GNG** (erreur accumulée, insertion) ; (h) les tests de vigilance **ART1 / ART2** ; (i) la perte du **VAE** ; (j) choisir la méthode adaptée à un objectif.
