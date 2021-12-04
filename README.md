# TS341 Détection des buts

Dans le cadre du projet d'outils d'imagerie pour la robotique, nous avons pour objectif la reconnaissance du but sur un terrain de foot dans le cadre de la Robocup.

## Chaine de traitement globale

Notre chaine de traitement prévisionnelle est composée de plusieurs modules. D'une part nous avons un réseau de neuronne permettant de détecter un but dans une image. D'autre part nous avons du traitement de l'image classique pour segmenter le but à travers cette image. Ce pipeline est résumé par la figure suivante.

![pipeline.png](readme_images/pipeline.png)

## Utilisation d'un réseau de neuronne pour trouver les buts

Pour ce faire, nous avons utilisé un algorithme R-CNN, déjà pré-entrainées sur le dataset COCO. R-CNN est une méthode de detection d'objets trés fiable et robuste, c'est pour cette raison que nous nous sommes tournées vers cet algorithme.
En entrée, il est donné à l'algorithme 579 images toutes labélisée par nos soins, issues des images données au début du projet. La moitié des images sont tournée de 20° afin d'augmenter le nombre d'image en entrée : 20% sont utilisée en train et 80% en test
RC
mobilenet  couche de convolution différente

En sortie de cette algorithme nous avons donc une image recadrée sur le but et la position (x,y) de(s) poteau(x) par rapport à l'image d'origine.

## Traitement de l'image pour segmenter les buts

L'entrée de cet algorithme récupère la sortie du précédent et a pour objectif de segmenter et reconstruire le but. Cette segmentation utilise du traitement de l'image classique plus précisément :

```
fonction segmentation_goal(image):
    # Conversion de l'image en noir et blanc en utilisant cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    gray_scale(img) 
    
    # Érosion de l'image
    erode(img, kernel = (2,2))
    
    # Application d'un filtre médian
    median(img, k = 5)
    
    # Détection des contours en utilisant Canny
    contours(img, 100, 200)
    
    # Détection des composantes connexes (cv2.connectedComponents) et selection du contour ayant l'aire la plus grande
    max_area_components(img)
    
    # Reconstruction du but 
    reconstruct_goal(img, (x_1, y_1), (x_2, y_2))
    
    # Supperposition du but sur l'image originale
    superpose(img, mask)
```

Un exemple d'éxécution de chaque étape est disponible dans le dossier readme_images.

![readme_grayscale.png](readme_images/readme_grayscale.png)
![readme_erode.png](readme_images/readme_erode.png)
![readme_median.png](readme_images/readme_median.png)
![readme_contours.png](readme_images/readme_contours.png)
![readme_maxcontour.png](readme_images/readme_maxcontour.png)
![readme_supperpose.png](readme_images/readme_supperpose.png)

Lors de l'écriture de cette milestone tous les algorithmes permettant le traitement d'image sont fonctionnels hormis reconstruct_goal. L'algorithme général est executable en réalisant la commande `python3 main.py`.