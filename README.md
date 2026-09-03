# compDEM

`compDEM` compare deux DEM photogrammétriques déjà alignés et détecte les changements significatifs de relief/profondeur.

Version actuelle : **4.5.6**. La géométrie de détection reste celle du profil **V4_GOLDEN** validé.

## Convention métier des sorties

Le moteur de détection conserve sa logique interne historique afin de ne pas modifier les détections. En revanche, les valeurs exportées suivent désormais la convention :

```text
valeur affichée = DEM_reference - DEM_compare

valeur > 0  -> change_type = "gain"
valeur < 0  -> change_type = "loss"
```

Ainsi, une ancienne valeur de `-11 mm` devient `+11 mm` et est exportée comme `gain`.

La couleur suit la même convention métier :

```text
gain / valeur positive -> bleu
loss / valeur négative -> rouge
|écart| <= seuil       -> transparent
```

## Installation

Python 3.10+ recommandé.

```bash
pip install -r requirements.txt
```

## Configuration

Exemple :

```json
{
  "reference_dem": "DEMInsp2.tif",
  "compare_dem": "DEMReinsp2-1.tif",
  "threshold_mm": 10.0,
  "output_prefix": "result",
  "output_dir": "./outputs"
}
```

Les deux DEM doivent avoir la même résolution et être alignés sur une grille compatible. Ils peuvent différer de quelques lignes ou colonnes : le programme travaille sur leur intersection géoréférencée, sans reprojection ni interpolation.

Les fichiers `.tfw` d'entrée peuvent être placés à côté des TIFF si nécessaire ; Rasterio/GDAL les lit automatiquement. Aucun `.tfw` de sortie n'est produit.

## Lancement

```bash
python compdem.py config.json
```

## Fichiers produits

```text
result_detections.geojson
result_zones.geojson
result_boxes.geojson
result_summary.json
result_difference.tif
result_difference_rgba.tif
```

### `result_boxes.geojson`

Chaque box finale contient notamment :

- `change_type` : `gain` ou `loss` ;
- `bbox_width_m` et `bbox_height_m` : dimensions de la box en mètres, à 0,001 m ;
- `center_y_m` : position monde Y du centre de la box, à 0,001 m ;
- `center_angle_deg` : position X du centre convertie linéairement sur 0° à 360° entre le bord gauche et le bord droit de l'emprise commune, à 0,1° ;
- `detected_pixel_count` : nombre de pixels réellement détectés ;
- `detected_area_m2` : surface de l'union des pixels réellement détectés ayant conduit à la box, et non surface du rectangle ;
- `median_depth_mm` : médiane signée des pixels détectés selon la convention `gain positif / loss négatif` ;
- `spatial_max_depth_mm` : maximum spatial signé selon la même convention.

Le maximum spatial est la plus forte amplitude encore supportée par une composante 8-connexe d'au moins **2 cm²**. Les signes opposés sont traités séparément.

### `result_difference.tif`

COG `float32` contenant la différence affichée :

```text
DEM_reference - DEM_compare
```

Paramètres principaux : blocs 512×512, compression DEFLATE, pyramides internes `NEAREST`, géoréférencement interne.

### `result_difference_rgba.tif`

Le nom est historique ; le fichier est un **COG RGB + masque interne** :

- blocs 512×512 ;
- RGB en JPEG qualité 95 / YCbCr ;
- pyramides RGB en JPEG qualité 95 ;
- pyramides en `NEAREST` ;
- masque de transparence interne GDAL ;
- gain positif : bleu ;
- loss négatif : rouge ;
- valeurs dans le seuil : transparentes.

Avec OpenLayers, utiliser `convertToRGB: 'auto'` ou `convertToRGB: true` pour le JPEG YCbCr. Le masque interne est utilisé comme transparence.

## Dépendances

- NumPy ;
- Rasterio/GDAL ;
- OpenCV.

## Version

**4.5.6** — inversion de la convention affichée : gain positif, loss négatif, ajout de `change_type`, inversion cohérente du raster de différence et de l'échelle colorée. Les pyramides COG restent en `NEAREST`.
