# TS341_Robocup

Dans le cadre du projet d'outils d'imagerie pour la robotique, nous avons pour objectif la reconnaissance du but sur un terrain de foot. Pour ce faire, nous avons utilisé un algorithme R-CNN, déjà pré-entrainées sur le dataset COCO. R-CNN est une méthode de detection d'objets trés fiable et robuste, c'est pour cette raison que nous nous sommes tournées vers cet algorithme.
En entrée, il est donné à l'algorithme 579 images toutes labélisée par nos soins, issues des images données au début du projet. La moitié des images sont tournée de 20° afin d'augmenter le nombre d'image en entrée : 20% sont utilisée en train et 80% en test
RC
mobilenet  couche de convolution différente
segmentation ==> Detection du but
