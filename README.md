# TS341 Détection des buts

Dans le cadre du projet d'outils d'imagerie pour la robotique, nous avons pour objectif la reconnaissance du but sur un terrain de foot dans le cadre de la Robocup.

Pour utiliser le projet il est aussi nécessaire d'avoir tensorflow 2.6 installé sur la machine. Tous les packages d'une environnement virtuel utilisé pour utiliser le projet sont présent dans `requirements.txt`.

Pour simplement l'utiliser il est nécessaire d'activer l'environnement virtuel et de décompresser `tod_tf2.zip`. Les résultats de l'entrainement du réseau sont dans [object_detection](tod_tf2/object_detection/training/ssd_mobilnet/saved_model3).

Ce projet nécessite d'avoir l'api TOD pour réentrainer le réseau pour l'installer : [ROS5PRO TOD](https://learn.e.ros4.pro/fr/vision/object_detection_tf2/tod_install/)

Pour simplement voir les résultats du projet : [process](process1)

Pour executer le projet sur une image : `python3 main.py photo <name>.png` où `name` est le nom d'une image dans le dossier `data`. L'image enregistrée sera dans le dossier `process`.

Pour executer le projet sur le dossier data : `python3 main.py folder`. Les images seront dans le dossier `process`.

Pour tout problème d'installation n'hésitez pas à nous contacter.

## Chaine de traitement globale

Notre chaine de traitement prévisionnelle est composée de plusieurs modules. D'une part nous avons un réseau de neurone permettant de détecter un but dans une image. D'autre part nous avons du traitement de l'image classique pour segmenter le but à travers cette image. L'utilisation de ce pipeline dans des conditions initiales est résumé dans la figure suivante.

![pipeline.png](readme_images/pipeline.png)

## Utilisation d'un réseau de neurone pour trouver les buts

Dans le cadre du projet d'outils d'imagerie pour la robotique, nous avons pour objectif la reconnaissance du but sur un terrain de foot. Pour ce faire, nous avons utilisé le deep learning pour la détection d'objet dans une image. Nous allons utiliser des réseaux de neurones déjà entrainés de la famille des réseaux convolutionnels : le MobilNet v2. Ces réseaux sont déjà pré-entrainées sur le dataset COCO. Le R-CNN est un algorithme de détection d'objet qui segmente l'image d'entrée pour trouver des zones délimitantes pertinentes, puis fait tourner un algorithme de détection pour trouver les objets les plus probables d'apparaître dans ces zones délimitantes. R-CNN est un réseau de neurones trés fiable et robuste, c'est pour cette raison que nous nous sommes tournées vers ce genre de réseaux. Il va permettre d'extraire des images les regions les plus susceptibles de contenir un objet(=zone d'interet). Pour chacune des zones d'intérêts, une boîte englobantes va être générés. Ces boites vont être classifiées et selectionnées en fonctions de leurs probabilités de contenir l'objet.   

En entrée, il est donné à l'algorithme 579 images toutes labellisées par nos soins, issues des images données au début du projet. La moitié des images sont tournée de 20° afin d'augmenter le nombre d'images en entrée : 20% sont utilisée en train (= permet d'entrainer le modèle) et 80% en test (= permet de mesurer l'erreur du modèle final sur des données inconnues). 

Un exemple d'image labélisée :  
![labelisation.png](readme_images/labelisation.png)
Nous avons décidé de détecter le bas des poteaux de but et non tout le but entier. En effet, sur certaines données il n'y a qu'un bout du but (1 poteau ou 2 poteaux sans le haut du but) mais il faut quand même détecter que cela est un but. Nous avons donc décidé de détecter le bas des poteaux puis nous recadrons la photo par rapport aux positions des boites englobantes détecté sur l'image. 

En sortie de cet algorithme, nous avons donc une image recadrée sur le but et la position (x,y) de(s) poteau(x) par rapport à l'image d'origine.  

L'entrainement de ce réseau avec notre dataset ce fera grâce à l'API TOD TF2 (qui utilise TensorFlow 2) que vous pouvez trouver [ici](https://github.com/cjlux/tod_tf2_tools).

## Traitement de l'image pour segmenter les buts

L'entrée de cet algorithme récupère la sortie du précédent et a pour objectif de segmenter et reconstruire le but. Cette segmentation utilise du traitement de l'image classique plus précisément. 

Dans premier temps l'image est convertie en niveau puis un seuillage est utilisé pour binariser l'image. 

![readme_grayscale.png](readme_images/readme_grayscale.png)

En binarisant l'image nous nous retrouvons avec le but d'une part mais aussi des artéfacts du décors qui bruitent l'image. Pour réduire ces bruits nous allons commencer par supprimer les artéfacts les plus petits en appliquant une erosion puis un filtre médian. 

![readme_median.png](readme_images/readme_median.png)

Après avoir éliminé les artéfacts de petite taille il reste encore isoler le but. Pour cela nous détectons les contours de l'image grâce à l'algorithme Canny. 

![readme_contours.png](readme_images/readme_contours.png)

Pour isoler le but de l'image nous allons chercher le contour ayant la plus grande aide possible, pour ce faire nous détectons tout d'abord les composantes connexes de l'image. Parmi ces composantes connexes nous choisissons la plus grande qui correspond à un but. 

![readme_maxcontour.png](readme_images/readme_maxcontour.png)

En supperposant cet aire avec l'image initiale nous avons une détection du périmètre du but

![readme_supperpose.png](readme_images/readme_supperpose.png)

La détection du périmètre du but a été testé sur 96 images et détecte correctement les contours dans 77% des cas. Dans 19% des cas le périmètre est partiellement détecté et dans 4% des cas l'algorithme échoue totalement.

### Reconstruction des lignes directrices

Une fois que le périmètre des buts est détecté nous souhaitons construire les lignes directrices des buts, nous réalisons ça mathématiquement en distinguant 3 cas. Dans tous les cas nous disposons de la base des poteaux grâce au réseau de neurones précédent.
1. **Un seul poteau** : dans ce cas nous prenons le périmètre détecté par le premier traitement, nous parcourons verticalement l'image jusqu'à trouver une densité de blanc suffisante (au moins 4 pixel blanc sur une ligne). Nous trouvons ensuite sur la ligne le début (parcours en partant de la gauche) et la fin (parcours en partant de la droite) du but. Nous trouvons alors le centre du poteau (en x) et sa hauteur (en y). Étant donné que nous avons la position d'origine du poteau il suffit de relier les deux

![post_reconstruct](readme_images/post_reconstruct.png)

2. **Deux poteaux** : le processus pour deux poteaux est le même que celui pour un, cependant lorsque deux poteaux sont à reconstruire on coupe l'image en deux puis on applique l'algorithme permettant de trouver un poteau aux deux moitiés du poteau. 
3. **Le but** : le but se reconstruit comme 2 poteaux mais la hauteur du but se détecte avec une densité de pixel blanc plus forte (plus de 50% de la ligne)

![goal_reconstruction](readme_images/5006.png)

Les algorithmes de reconstructions ont été testés sur les mêmes 96 images que précedemment. Ceux ci donnent des résultats mitigés : 40.6% des buts sont correctement reconstruits, 9.4% le sont partiellement, 16.7% conduisent à des abérations et dans 33.3% des cas l'algorithme n'essaye pas de reconstruire le but faute de réussite. 

Ces résultats mitigés sont le fruit de la méthode mathématiques employée. En effet cette méthode est particulièrement efficace pour reconstruire les buts en entier mais dépend énormément de la qualité du périmètre détecté. Une voie d'amélioration de cette parte serait de détecter soit les coins du buts pour en construire les lignes directrices soit d'appliquer une transformée de Hough pour trouver les lignes directrices et ensuite déterminer le but.