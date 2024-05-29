# Projet Waterflow

**SUJET :**  

L'eau est essentiel à la vie cependant il est extremement delicat si ce n'est impossible de determiner sa potabilité par une simple analyse visuel de cette dernière. 

Ainsi entouré de 2 collaborateurs, nous decidons de creer des modèles de ML afin de determiner la potabilité de l'eau à partir de différentes données.
Au cours de ce projet nous avons tenté trois approches différentes vis à vis du dataset mis à notre disposition.

# Dataset

Nous avons un dataset à notre disposition qui nous semble au premier coup d'oeil assez complet et avec des données pertinente compte tenu de l'objectifs de notre projet.
Cependant suite à une premieère analyse des données et à l'étude des normes internationnale concernant la potabilité de l'eau nous nous rendons compte que le dataset semble bancale.

Il y a plusieurs explication à ceci selon nous :
-Un mauvais étalonnage des capteurs
-Un mauvais protocole concernant la prise de mesure
-Une erreur humaine lors de la saisi ou de la transmission des données
-Une anonymisation des données qui rend le dataset quelque peu erroné sur certains points

Nous nous sommes donc rendu compte que la labélisation est erroné compte tenu des normes sur la potabilité de l'eau.
De plus différentes mesures ayant des liens physiques entre elle nous montre que les données semblent erronées.


Ainsi face à cette problématique nous nous sommes orientés sur différentes approches.


### Utilisation de l'ensemble du dataset sans modification :

Dans un premier temps nous avons tenté d'entrainer nos premiers modèles sur les données brut avec les labels initiaux.
Suite à un preprocessing classique pour préparer les données nous avons entrainé de nos modèles.
Les resultats de ces derniers se rapprochent fortement de l'aléatoire comme nous le présupposions vis à vis de nos précedentes analysés des données.

### Utilisation de l'ensemble du dataset avec relabelisation :

Dans un second temps nous avons tenté d'entrainer nos modèles sur les données relabelisé par nos soins en appliquant les différentes normes internationnal concernant la potabilité de l'eau.
Suite au même preprocessing classique pour préparer les données nous avons entrainé de nos modèles.
Lors de cette étape nous obtenons de meilleurs résultat même si les incohérences dans les données doivent pénaliser le score de ces derniers.

### Utilisation d'une partie du dataset avec relabelisation :

Dans un dernier temps nous avons tenté d'entrainer nos modèles sur les données relabelisé par nos soins en appliquant les différentes normes internationnal concernant la potabilité de l'eau.
Nous avons modifier le preprocessing pour préparer les données et selectionner les features les plus importantes.
La selection s'est faite selon notre analyse et non selon l'utilisation d'un algorithme, nous avons retenus le PH, la turbidité, les sulphates ainsi que la dureté de l'eau.
Suite à cela nous avons entrainé de nos modèles.
Lors de cette étape nous obtenons encore de meilleurs résultat.



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
    * precision : 0.86

* voting classifier (RF + LR) :  
    * une accuracy de 0.62 sur les données avec les labels originaux.
    * precision : 0.76
 
      
* Logistic regression :  
    * une accuracy de 0.61 sur les données avec les labels originaux.
    * precision : 0.76
    
Nous décidons donc d'utiliser le Random forest pour notre application.

        


