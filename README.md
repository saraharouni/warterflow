# Projet Waterflow

**SUJET :**  

L'eau est essentielle à la vie cependant il est extrêmement délicat si ce n'est impossible de déterminer sa potabilité par une simple analyse visuel de cette dernière. 

Ainsi entouré de 2 collaborateurs, nous décidons de créer des modèles de ML afin de déterminer la potabilité de l'eau à partir de différentes données.
Au cours de ce projet nous avons tenté trois approches différentes vis à vis du dataset mis à notre disposition.

# Dataset

Nous avons un dataset à notre disposition qui nous semble au premier coup d'œil assez complet et avec des données pertinentes compte tenu de l'objectifs de notre projet.
Cependant à la suite d’une première analyse des données et à l'étude des normes internationale concernant la potabilité de l'eau nous nous rendons compte que le dataset semble bancale.

Il y a plusieurs explications à ceci selon nous :  

-Un mauvais étalonnage des capteurs
-Un mauvais protocole concernant la prise de mesure
-Une erreur humaine lors de la saisi ou de la transmission des données
-Une anonymisation des données qui rend le dataset quelque peu erroné sur certains points

Nous nous sommes donc rendu compte que la labélisation est erronée compte tenu des normes sur la potabilité de l'eau.
De plus différentes mesures ayant des liens physiques entre elle nous montre que les données semblent erronées.


Ainsi face à cette problématique nous nous sommes orientés sur différentes approches.


### Utilisation de l'ensemble du dataset sans modification :

Dans un premier temps nous avons tenté d'entrainer nos premiers modèles sur les données brut avec les labels initiaux.
À la suite d’un preprocessing classique pour préparer les données nous avons entrainé de nos modèles.
Les résultats de ces derniers se rapprochent fortement de l'aléatoire comme nous le présupposions vis à vis de nos précédents analyses des données.

### Utilisation de l'ensemble du dataset avec relabelisation :

Dans un second temps nous avons tenté d'entrainer nos modèles sur les données relabelisé par nos soins en appliquant les différentes normes internationales concernant la potabilité de l'eau.
À la suite du même preprocessing classique pour préparer les données nous avons entrainé de nos modèles.
Lors de cette étape nous obtenons de meilleur résultat même si les incohérences dans les données doivent pénaliser le score de ces derniers.

### Utilisation d'une partie du dataset avec relabelisation :

Dans un dernier temps nous avons tenté d'entrainer nos modèles sur les données relabelisé par nos soins en appliquant les différentes normes internationales concernant la potabilité de l'eau.
Nous avons modifié le preprocessing pour préparer les données et sélectionner les features les plus importantes.
La sélection s'est faite selon notre analyse et non selon l'utilisation d'un algorithme, nous avons retenus le PH, la turbidité, les sulfates ainsi que la dureté de l'eau.
Suite à cela nous avons entrainé de nos modèles.
Lors de cette étape nous obtenons encore de meilleur résultat.



# Procédure

* Faire une analyse exploratoire des données
* Faire une veille des normes concernant la potabilité
* Concevoir plusieurs modèles sur MLFlow
* Créer une application Flask qui fera les prédictions

# Application

Cette application permet de faire des prédictions sur la potabilité d'une eau. 
Pour l'utiliser, il suffit de répondre aux questions et de cliquer sur le bouton *prédiction*


# Conclusion :
Nous avons utilisé plusieurs modèles de classification pour estimer la potabilité d'une eau :  

* Random forest :
    * une accuracy de 0.85 sur les données avec les labels originaux.
    * precision: 0.86

* voting classifier (RF + LR) :  
    * une accuracy de 0.62 sur les données avec les labels originaux.
    * précision : 0.76
 
      
* Logistic regression :  
    * une accuracy de 0.61 sur les données avec les labels originaux.
    * précision : 0.76
    
Nous décidons donc d'utiliser le Random forest pour notre application.
