# compDEM

`compDEM` compare deux DEM photogrammétriques déjà alignés et détecte les changements significatifs de relief/profondeur.

La version actuelle correspond au profil validé **V4.2 STATS**. L'algorithme métier est volontairement calibré dans le code ; l'utilisateur ne règle que les entrées et le seuil vertical.

## Principe

Le programme :

1. ouvre les deux DEM GeoTIFF ;
2. travaille uniquement sur l'intersection de leurs emprises géoréférencées ;
3. calcule `DEM_compare - DEM_reference` sans rééchantillonnage ;
4. détecte les zones compactes/irrégulières par densité spatiale ;
5. détecte les lignes fortes et fragmentées par analyse de direction (PCA) ;
6. regroupe les zones proches et fusionne les bounding boxes qui se chevauchent ;
7. exporte les résultats vectoriels et raster.

Les deux DEM peuvent différer de quelques lignes ou colonnes : l'analyse se fait sur leur intersection dans le repère monde.

## Installation

Python 3.10+ recommandé.

```bash
pip install -r requirements.txt
```

## Configuration

Copier `config.example.json` en `config.json` puis renseigner les deux DEM :

```json
{
  "reference_dem": "DEM_reference.tif",
  "compare_dem": "DEM_compare.tif",
  "threshold_mm": 10.0,
  "output_prefix": "result"
}
```

Les fichiers `.tfw`, s'ils sont utilisés, doivent être placés à côté des TIFF avec le même nom de base. Rasterio/GDAL les lit automatiquement.

`threshold_mm` est le seuil vertical : avec `10.0`, les variations comprises entre -10 mm et +10 mm ne sont pas considérées comme du signal utile.

Un champ optionnel `output_dir` peut être ajouté pour choisir le dossier de sortie :

```json
"output_dir": "./outputs"
```

## Lancement

```bash
python compdem.py config.json
```

La différence calculée est toujours :

```text
DEM_compare - DEM_reference
```

## Fichiers produits

Pour `"output_prefix": "result"` :

```text
result_detections.geojson
result_zones.geojson
result_boxes.geojson
result_summary.json
result_difference.tif
result_difference.tfw
result_difference_rgba.tif
result_difference_rgba.tfw
```

### `result_boxes.geojson`

C'est la sortie principale pour les zones détectées. Chaque bounding box contient notamment :

- le signe du changement ;
- ses dimensions ;
- le nombre de pixels réellement détectés ;
- `median_depth_mm` : profondeur médiane absolue des pixels détectés ;
- `max_depth_mm` : profondeur maximale absolue des pixels détectés ;
- `median_dz_mm` : médiane signée du changement.

Les statistiques de profondeur sont calculées uniquement sur les pixels appartenant aux détections avérées, et non sur toute la surface rectangulaire de la box.

### `result_difference.tif`

GeoTIFF `float32` contenant la différence brute entre les deux DEM.

### `result_difference_rgba.tif`

GeoTIFF RGBA destiné à la visualisation :

- `|ΔZ| <= threshold_mm` : totalement transparent ;
- `ΔZ > threshold_mm` : rouge vif ;
- `ΔZ < -threshold_mm` : bleu vif.

Le canal alpha est réel. Ce raster peut donc être tuilé en PNG transparent pour un affichage web, notamment avec OpenLayers.

## Dépendances

- NumPy : calcul matriciel ;
- Rasterio/GDAL : lecture/écriture et géoréférencement des rasters ;
- OpenCV : morphologie, densité spatiale et composantes connexes.

## Version

`4.2.0` — profil V4_GOLDEN refactorisé, exports raster RGBA et statistiques de profondeur par bounding box.
