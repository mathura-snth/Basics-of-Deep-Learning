
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

## Exemple : le XOR (diapos 28–29)

Données (codage −1/+1) : (1,1)→−1 ; (1,−1)→+1 ; (−1,1)→+1 ; (−1,−1)→−1. Un perceptron simple ne peut pas séparer ces classes (non linéairement séparables).

**Idée du réseau de la diapo :** deux unités cachées définissent **deux droites parallèles** (ex. x₁+x₂+0,5 = 0 et x₁+x₂+1,5 = 0 sur la diapo). La bande entre les deux droites contient une classe, l'extérieur l'autre ; la sortie combine les deux unités cachées (poids 1,0 et −1,0, biais 0,5).

Construction équivalente que vous pouvez vérifier à la main : h₁ = signe(x₁+x₂+1), h₂ = signe(x₁+x₂−1), sortie = signe(h₁ − h₂ − 1).

- (1,1) : h₁=+1, h₂=+1 → 0−1 <0 → −1 ✓
- (1,−1) et (−1,1) : h₁=+1, h₂=−1 → 2−1 >0 → +1 ✓
- (−1,−1) : h₁=−1, h₂=−1 → 0−1 <0 → −1 ✓

**Leçon :** une couche cachée transforme l'espace d'entrée en un espace où le problème devient linéairement séparable.

**Exercice diapo 36** (à refaire seul) : réseau 2 entrées – 2 cachées – 2 sorties avec ReLU et les poids affichés. Procédure : (1) passe avant pour un exemple (calculer a, ReLU, puis y) ; (2) δ de sortie = (y−d)·f'(a) avec f'=1 si a>0 sinon 0 ; (3) δ cachés = f'(a\_j)·Σ w\_hj δ\_h ; (4) Δw = −ε δ\_j s\_i.

---

## 10. Organisation des données (diapos 30–36)

### Codage des sorties (classification à C classes)

| Codage | Bleu | Blanc | Rouge |
| --- | --- | --- | --- |
| **1-parmi-C** (0/1) | 1 0 0 | 0 1 0 | 0 0 1 |
| **1-parmi-C** (−1/1) | 1 −1 −1 | −1 1 −1 | −1 −1 1 |
| **1-parmi-(C−1)** (0/1) | 1 0 | 0 1 | 0 0 |
| **Thermomètre** (0/1) | 1 1 1 | 0 1 1 | 0 0 1 |

- 1-parmi-C : une sortie par classe, la classe prédite est la sortie maximale.
- 1-parmi-(C−1) : une sortie de moins (la dernière classe = tout à 0).
- **Thermomètre** : à utiliser quand les classes sont **ordonnées** (classement de classes).

### Normalisation des entrées

Sans normalisation, les variables à grande échelle (ex. altitude en milliers vs pente en dizaines) dominent. Résultat illustré diapo 32 : sans normalisation la précision stagne, avec elle l'apprentissage converge.

Centrer-réduire :

```
x(n) ← ( x(n) − x̄ ) / σ,    x̄ = (1/N) Σ_n x(n),    σ² = (1/N) Σ_n ( x(n) − x̄ )²
```

Pourquoi : (1) les entrées centrées-réduites évitent de **saturer** les sigmoïdes (zones où f' ≈ 0) ; (2) la surface d'erreur est plus « ronde » : quand les paramètres ont des échelles différentes, le gradient du plus grand domine la mise à jour (ellipses allongées, trajectoire en zigzag) ; avec des échelles égales, tous les paramètres sont mis à jour dans des proportions comparables.

### Initialisation des poids

Objectif : que l'activation `a_i = Σ_{j∈Amont(i)} w_ij x_j` ait une variance ≈ 1 pour ne pas saturer la sigmoïde dès le départ.

Hypothèses : entrées centrées-réduites (E\[x\]=0, Var\[x\]=1), poids indépendants des entrées et de moyenne nulle (E\[w\]=0).

```
E[a_i]   = Σ_j E[w_ij] E[x_j] = 0
Var[a_i] = Σ_j E[w_ij²] E[x_j²] = |Amont(i)| · σ²[w_ij]
```

On veut `Var[a_i] = 1`, donc :

```
σ²[w_ij] = 1 / |Amont(i)|     ⟹     σ[w_ij] ∝ |Amont(i)|^(−1/2)
```

On tire donc les poids dans une loi uniforme :

```
w_ij ∈ [ −k/√|Amont(i)| , +k/√|Amont(i)| ],   0,5 < k < 2
```

Avec la fonction recommandée `f(x) = 1,71·tanh(2x/3)`. Moins il y a d'entrées, plus les poids peuvent être grands, et inversement. *Ne jamais initialiser tous les poids à 0 ou à la même valeur* : toutes les unités cachées resteraient identiques (symétrie).

---

## 11. Approximation universelle et profondeur (diapos 37–43)

- **Construction des diapos 40** : une sigmoïde seule = une « marche » dans une direction ; la somme de 2 sigmoïdes = une **crête** (ridge) ; la somme de 2 crêtes = une fonction avec un **maximum** ; une sigmoïde appliquée dessus donne une **réponse localisée** (« bosse »). En combinant linéairement ces bosses, on approche toute fonction continue.
- **Théorème de Hornik (1989)** : un réseau à trois couches (une cachée) avec sorties linéaires et un nombre suffisant d'unités cachées saturantes croissantes peut approcher arbitrairement toute fonction mesurable bornée d'un espace de dimension finie vers un autre. **Un RNA est un approximateur universel.**
- **Limites** : théorème **purement existentiel** — il ne dit ni combien de neurones cacher, ni comment trouver les poids, ni que l'apprentissage convergera. En pratique, plusieurs couches et sorties non linéaires peuvent être plus commodes.
- **Intérêt de la profondeur** (GINN, diapos 41–42) : un réseau à 3 couches cachées de 16 neurones apprend à classer chaque pixel (x,y) d'une image (noir/blanc) ; les couches successives plient et déforment l'espace pour produire des frontières complexes avec peu d'unités. Un réseau profond est généralement plus compact qu'un réseau peu profond mais très large.

---

## 12. Liens avec l'analyse de données (diapos 44–47)

### MLP et ACP (PCA)

Un **auto-encodeur** (réseau auto-associatif) est entraîné à reproduire son entrée : `d = x`. Architecture : entrée n → **couche de codage** (peu d'unités) → sortie n ; l'**encodeur** compresse, le **décodeur** reconstruit. Avec des sorties **linéaires**, il réalise une transformation de **Karhunen-Loève** : il est équivalent à une **ACP** (Bourlard & Kamp 1988 ; Baldi & Hornik 1989) — la couche cachée engendre le même sous-espace que les premières composantes principales.

Avec des activations **non linéaires**, l'auto-encodeur capte des structures que l'ACP linéaire ne voit pas (diapos 45–46 : sur MNIST ou des données textuelles, les classes sont mieux séparées dans l'espace du codage que sur les 2 premières composantes).

### MLP et LDA (analyse discriminante)

Gallinari et al. (1991) : un MLP entraîné pour classer effectue une analyse discriminante ; à chaque couche cachée, les classes sont de plus en plus séparées (diapo 47 : entrée → cachée 1 → cachée 2 → sortie, les nuages de classes se séparent progressivement).

---

## 13. Réseaux structurés (diapos 48–60)

Idée générale : **contraindre l'architecture** (connexions locales, poids partagés, récurrence) pour réduire le nombre de paramètres et injecter de la connaissance sur le problème.

### Réseau complètement connecté (diapo 48)

Chaque unité d'une couche est reliée à toutes celles de la suivante. Beaucoup de paramètres, aucune structure imposée.

### Réseaux récurrents simples avec contexte

- **Elman (1990)** : les sorties de la couche **cachée** à l'instant t−1 sont recopiées dans des **unités de contexte** `c(t−1)` qui servent d'entrées supplémentaires à t. Mémoire du passé interne.
- **Jordan (1992)** : ce sont les **sorties du réseau** `y(t−1)` qui sont réinjectées en contexte.
- **NETtalk (Sejnowski & Rosenberg, 1987)** : synthèse de parole (texte → phonèmes). Le réseau regarde une **fenêtre glissante de lettres** (la lettre centrale + son contexte) et prédit le phonème de la lettre centrale.

### Réseaux récurrents (RNN) (diapos 52–54)

Un RNN a une **boucle** : l'information persiste d'un pas de temps au suivant.

```
h_t = f_W( h_{t−1}, x_t )         (nouvel état = fonction de l'ancien état et de l'entrée)
y_t = f( h_t )                    (sortie au temps t)
```

Mêmes poids W à chaque pas de temps. **Dépliage (unrolling)** : un RNN = plusieurs copies du même réseau, chacune transmettant un message à la suivante.

Avantages : traite des **séquences de longueur variable** ; la complexité (nb de paramètres) ne dépend pas de la longueur ; poids réutilisés. **Inconvénient majeur** : difficultés d'entraînement dues aux **gradients explosifs et évanescents** (la rétropropagation dans le temps multiplie à répétition des dérivées/matrices : si leur norme est < 1 le gradient disparaît, si > 1 il explose). Le RNN opère de façon « **multiplicative** » (tanh répété).

### LSTM et GRU (diapos 55–57)

Deux cellules « à portes », qui traitent l'information de façon **additive** : **LSTM** (Hochreiter & Schmidhuber 1997) et **GRU** (Cho et al. 2014). Une **porte sigmoïde** sort des nombres entre 0 et 1 : la proportion de chaque composante qui « passe » (0 = rien, 1 = tout). Ex. `f_t = σ(W_f·[h_{t−1}, x_t] + b_f)`.

Le cœur du LSTM est l'**état de cellule C\_t** : mémoire à long terme, qui traverse les pas de temps avec peu d'interactions linéaires, d'où un **flux de gradient ininterrompu** (les erreurs passées continuent d'influencer l'apprentissage). Trois portes :

```
Porte d'oubli (forget) :  f_t = σ( W_f·[h_{t−1}, x_t] + b_f )     → quelle part de C_{t−1} on garde
Porte d'entrée (input) :  i_t = σ( W_i·[h_{t−1}, x_t] + b_i )     → quelle part de la nouveauté on accepte
Candidat mémoire       :  C̃_t = tanh( W_C·[h_{t−1}, x_t] + b_C )
Mise à jour de l'état  :  C_t = f_t ∗ C_{t−1} + i_t ∗ C̃_t           (∗ = produit terme à terme)
Porte de sortie        :  o_t = σ( W_o·[h_{t−1}, x_t] + b_o )
Sortie                 :  h_t = o_t ∗ tanh( C_t )
```

La mise à jour **additive** de C\_t est la clé contre le gradient évanescent.

### Connexions locales et poids partagés (diapos 58–60)

- **Connexions locales** : chaque unité cachée ne « voit » qu'un **champ récepteur** (petite zone de l'entrée), ce qui réduit fortement le nombre de poids.
- **Poids partagés (contraints)** : le **même filtre** (noyau de convolution) est appliqué à toutes les positions → des **extracteurs de caractéristiques** (feature extractors) qui détectent la même caractéristique locale partout. Très peu de paramètres libres.
- **TDNN** (Time Delay Neural Network, Waibel et al. 1987) : même principe appliqué à l'axe **temps** pour la reconnaissance de phonèmes sur un spectrogramme (fréquence × temps) ; le nombre de positions est environ `(N − d)/s + 1` pour une fenêtre d, un pas s.

---

## 14. Réseaux convolutifs (CNN) (diapos 61–92)

### Architecture type (LeNet, diapo 62)

Entrée (32×32) → **convolutions** (cartes de caractéristiques 6@28×28) → **sous-échantillonnage / pooling** (6@14×14) → convolutions (16@10×10) → pooling (16@5×5) → couches **entièrement connectées** (120, 84) → sortie (10 classes). Alternance détection de motifs / réduction de résolution.

### La convolution discrète (diapos 71–72)

Transformation **linéaire**, qui préserve l'ordre (la structure spatiale), **parcimonieuse** (quelques entrées seulement contribuent à chaque sortie) et qui **réutilise les paramètres** (mêmes poids en toutes positions).

Calcul : on **fait glisser** le noyau sur la carte d'entrée ; à chaque position, on multiplie terme à terme noyau et zone recouverte, puis on **somme**.

Exemple diapo 71 : image 5×5 (1ʳᵉ zone 3×3 : `1 1 1 / 0 1 1 / 0 0 1`), noyau 3×3 avec des 1 aux coins et au centre, 0 ailleurs (`1 0 1 / 0 1 0 / 1 0 1`) :

```
1·1 + 1·0 + 1·1 + 0·0 + 1·1 + 1·0 + 0·1 + 0·0 + 1·1 = 4
```

→ première case de la « convolved feature » = **4**. On décale ensuite le noyau et on recommence.

Définitions : **stride (pas)** = distance entre deux positions consécutives du noyau ; **zero padding** = nombre de zéros ajoutés au début et à la fin de chaque axe.

### Formules de taille de sortie (diapos 79–84)

Notations : i = taille d'entrée, k = taille du noyau, s = stride, p = padding, o = taille de sortie (entrées/noyaux carrés).

| Relation | Cas | Formule | Exemple |
| --- | --- | --- | --- |
| 1 | pas de padding, s=1 | o = (i−k)+1 | i=4, k=3 → o=2 |
| 2 | padding p, s=1 | o = (i−k)+2p+1 | i=5, k=4, p=2 → o=6 |
| 3 | **half/same** : k impair, p=⌊k/2⌋, s=1 | o = i | i=5, k=3, p=1 → o=5 |
| 4 | **full** : p=k−1, s=1 | o = i + (k−1) | i=5, k=3, p=2 → o=7 |
| 5 | p=0, stride s | o = ⌊(i−k)/s⌋+1 |  |
| 6 | **général** | **o = ⌊(i+2p−k)/s⌋ + 1** |  |

À retenir : la relation 6 contient toutes les autres. Le plancher ⌊·⌋ implique qu'**plusieurs tailles d'entrée peuvent donner la même sortie** quand s>1 (certaines lignes/colonnes en bordure ne sont pas couvertes par le noyau).

### Pooling (diapos 73–78)

On fait glisser une fenêtre et on applique une fonction de **résumé** (comme une convolution, mais non linéaire / pas de noyau appris) :

- **Max pooling** : le maximum de la fenêtre. **Average pooling** : la moyenne.
- Fenêtre 2×2 courante : élimine **3/4 des activations**.
- Rôle : réduire la résolution de la carte tout en gardant les caractéristiques utiles → **invariance aux petites translations** (et rotations légères), moins de calculs, moins de sur-apprentissage.

Exemple (diapo 77) : entrée 4×4 `[12 20 30 0 / 8 12 2 0 / 34 70 37 4 / 112 100 25 12]`, fenêtre 2×2, stride 2 :

- max pooling → `[20 30 / 112 37]` ;
- average pooling → `[13 8 / 79 20]` (ex. (12+20+8+12)/4 = 13 ; (37+4+25+12)/4 = 19,5 ≈ 20).

Variantes : **stochastique** (une activation tirée au hasard dans chaque région, effet de régularisation), **avec recouvrement** (fenêtres qui se chevauchent), **fractionnaire** (le taux de réduction peut être non entier, ce qui permet des réseaux plus profonds).

### Convolution transposée (« déconvolution ») (diapos 85–88)

Opération qui va **dans le sens inverse** d'une convolution : elle part d'une carte ayant la forme de la sortie d'une convolution et reconstruit une carte ayant la forme de son entrée, avec un motif de connexions compatible. Utilisations : **couche de décodage d'un auto-encodeur convolutif**, projection de cartes vers un espace de plus grande dimension (super-résolution, segmentation).

**Représentation matricielle** : en déroulant entrée et sortie en vecteurs, la convolution est une multiplication par une **matrice creuse C** dont les éléments non nuls sont les poids `w_{i,j}` du noyau. Pour une entrée 4×4 (vecteur de dimension 16) et un noyau 3×3 : C est de taille **4×16** et donne un vecteur de dimension 4 (sortie 2×2).

- Passe avant : `y = C·x`.
- Rétropropagation de l'erreur : multiplication par **Cᵀ** (16×4) : 4 dimensions → 16 dimensions.
- La **transposée** = on échange passe avant et passe arrière ; le même noyau w définit C et Cᵀ.

**Relation 8** : une convolution (s=1, p=0, noyau k) a pour transposée une convolution avec k'=k, s'=s, p'=k−1, et taille de sortie `o' = i' + (k−1)`. Elle équivaut à une convolution **full padding**. Exemple : convolution 4×4 → 2×2 ; sa transposée : 2×2 → 4×4.

### Exemples applicatifs du cours

- **LeNet** (LeCun et al. 1989) : reconnaissance de chiffres manuscrits (codes postaux) par rétropropagation sur un réseau convolutif.
- **Identification du locuteur** (Bennani & Gallinari 1991) : un TDNN extrait des caractéristiques sur le spectrogramme ; un **HMM par locuteur** utilise ces caractéristiques.
- **Segmentation médicale** : encodeur convolutif (Convolution + Batch Normalization + ReLU, pooling) puis décodeur (déconvolution, 1×1 convolution) → carte de segmentation de même taille que l'image.
- **Réseau « ver nématode »** (diapo 70) : un réseau inspiré du système nerveux du ver pilote une voiture avec seulement 19 neurones de contrôle, après des couches convolutives de perception.

### Super-résolution (SRCNN / FSRCNN) (diapos 63–67) : comptage des paramètres

Objectif : passer d'une image basse résolution (LR) à haute résolution (HR).

**SRCNN** (3 couches de convolution) :

1. Extraction de patchs : 64 filtres 1×9×9, ReLU → poids `1×9×9×64 = 5184` + 64 biais.
2. Mapping non linéaire : 32 filtres 64×1×1, ReLU → `64×1×1×32 = 2048` + 32 biais.
3. Reconstruction : 1 filtre 32×5×5, identité → `32×5×5×1 = 800` + 1 biais.

*Règle générale :* nombre de poids d'une couche convolutive = `(nb filtres) × (canaux d'entrée) × (hauteur) × (largeur)` ; biais = nb de filtres.

**FSRCNN** (plus compact, 5 étapes) : (1) *extraction* (56 filtres 1×5×5 : 1400 poids) ; (2) *shrinking* (12 filtres 56×1×1 : 672 poids) qui réduit la dimension des vecteurs de caractéristiques ; (3) *mapping non linéaire* (4 couches de 12 filtres 12×3×3 : 4×12×3×3×12 = 5184 poids) ; (4) *expanding* (56 filtres 12×1×1 : 672 poids) qui fait l'opération inverse du shrinking ; (5) *déconvolution* (un filtre 56×9×9 : 4536 poids) qui produit l'image HR. Total ≈ **12 464 poids** (+ quelques paramètres de PReLU).

---

## 15. Deep learning : objectifs et difficultés (diapos 93–94)

**Objectif** : **construire un nouvel espace de représentation** par des architectures profondes (comme les couches cachées d'un MLP, mais en plusieurs niveaux d'abstraction). L'apprentissage non supervisé entre les couches peut décomposer le problème en sous-problèmes distribués, de plus en plus abstraits.

**Difficultés de l'apprentissage supervisé de réseaux profonds :**

- Les premières couches sont mal apprises : **diffusion du gradient** — l'erreur diminue en se propageant vers les couches précédentes, donc apprentissage très lent. Plus le réseau est profond, plus la non-linéarité est forte, ce qui augmente les risques d'obstacles à l'optimisation ; les couches basses restent avec des transformations peu utiles.
- **Pas assez de données étiquetées** : on cherche des approches non supervisées, semi-supervisées ou par **transfert** pour exploiter les données non étiquetées.
- Davantage de **minima locaux** que dans les réseaux peu profonds.

**Réponses vues dans le cours** : ReLU (gradient mieux propagé), bonne initialisation et normalisation des données, cellules à portes (LSTM/GRU), poids partagés (CNN), pré-apprentissage non supervisé (auto-encodeurs).

---

## 16. Checklist de révision

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
