# Projet_AI_forgames

![Preview](/preview.png)
<br>
<br>

## 1. Introduction et objectif du projet

Le projet **Dino Game IA+** est un mini-jeu développé en Python avec la bibliothèque **Pygame**, inspiré du célèbre jeu du dinosaure de Google Chrome. L’objectif principal du projet est d’intégrer différentes formes d’intelligence artificielle dans un environnement ludique simple afin de démontrer la compréhension et l’application des concepts d’IA dans le cadre du développement de jeux.

Ce projet vise à montrer comment l’intelligence artificielle peut influencer le gameplay, adapter la difficulté et interagir avec le joueur de manière dynamique, tout en conservant une expérience de jeu fluide et agréable.

---

## 2. Description générale du jeu

Le jeu reprend le principe classique du Dino Game :

* Le dinosaure court automatiquement.
* Des obstacles apparaissent (cactus et oiseaux).
* Le joueur peut sauter ou s’accroupir pour éviter les obstacles.
* Le score augmente avec la distance parcourue.

À cette base, plusieurs couches d’intelligence artificielle ont été ajoutées afin d’enrichir le gameplay.

---

## 3. Rôle et types d’intelligence artificielle

### 3.1 IA concurrente (IA décisionnelle)

Une IA concurrente contrôle un deuxième dinosaure qui joue en parallèle avec le joueur. Cette IA observe la distance entre le dinosaure et les obstacles et décide quand sauter.

**Caractéristiques :**

* Prise de décision en temps réel.
* Réaction basée sur la distance à l’obstacle et la vitesse du jeu.
* Ajout d’une légère part d’aléatoire pour éviter un comportement parfait.

**Type d’IA :** IA décisionnelle basée sur des règles.

---

### 3.2 Boss IA adaptatif (Decision Tree)

Le Boss IA est responsable de la génération des obstacles. Son rôle est d’adapter la difficulté du jeu en fonction du comportement du joueur.

Le Boss IA utilise un **Decision Tree (arbre de décision)** entraîné sur un petit dataset représentant différents styles de joueurs.

**Variables d’entrée du modèle :**

* Taux de saut du joueur (jump rate)
* Taux d’accroupissement (duck rate)
* Score actuel

**Sortie du modèle :**

* Type d’obstacle à générer : petit cactus, grand cactus ou oiseau.

Le modèle permet au boss de choisir dynamiquement les obstacles les plus pertinents selon le style de jeu observé.

---

## 4. Problème rencontré et solution apportée

Lors de l’intégration du Decision Tree, un problème de gameplay est apparu : le modèle devenait trop optimal et rendait le jeu prévisible (enchaînement répétitif de sauts et d’accroupissements).

Pour résoudre ce problème, une **couche de régulation gameplay** a été ajoutée entre le modèle et le jeu :

* Introduction d’une probabilité d’obstacle aléatoire.
* Mémoire courte empêchant la répétition excessive du même type d’obstacle.

Cette approche permet de conserver l’intelligence adaptative du modèle tout en garantissant une expérience de jeu plus naturelle et plus amusante.

---

## 5. Prototype technique

Le prototype est entièrement fonctionnel et comprend :

* Une interface graphique jouable.
* Un système de collision.
* Un score dynamique.
* Une IA concurrente visible à l’écran.
* Un Boss IA adaptatif basé sur un modèle d’apprentissage.

Le code est structuré en classes distinctes afin de séparer clairement le gameplay et la logique IA.

---

## 6. Plan d’évolution du projet

Les évolutions possibles du projet incluent :

* Enrichissement du dataset du Decision Tree avec des données réelles collectées pendant les parties.
* Réentraînement dynamique du modèle.
* Ajout d’une IA prédictive capable d’anticiper les mouvements futurs du joueur.
* Amélioration de l’interface graphique et des animations.

---

## 7. Conclusion

Le projet **Dino Game IA+** démontre comment différentes approches d’intelligence artificielle (décisionnelle et adaptative) peuvent être intégrées efficacement dans un mini-jeu. L’utilisation d’un Decision Tree permet d’adapter le gameplay au comportement du joueur, tout en respectant les contraintes de fun et de jouabilité grâce à une régulation intelligente.

Ce projet répond aux objectifs pédagogiques du module *AI Programming for Games* en proposant une application concrète, fonctionnelle et évolutive de l’intelligence artificielle dans un contexte ludique.

