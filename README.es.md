# Detección semi-supervisada con detectores de vocabulario abierto sobre xView

[Galego](README.md) · [English](README.en.md) · **Español**

Código del Trabajo de Fin de Grado sobre **detección de objetos semi-supervisada (SSOD)** en
imagen satelital (conjunto de datos **xView**), que integra los detectores de vocabulario abierto
**Grounding DINO** y **Rex-Omni** congelados como fuente complementaria de pseudoetiquetas sobre
un detector base **Faster R-CNN**.

Todo el flujo está definido como un **pipeline reproducible con [DVC](https://dvc.org)** en
`dvc.yaml`, parametrizado en `params.yaml`. Este repositorio contiene el **código** y las **métricas de resultados** (ficheros JSON pequeños);
los datos, las salidas intermedias voluminosas y los pesos no se incluyen, y se obtienen como se
describe en la [Sección 2](#2-obtener-los-datos).

---

## Visión general del pipeline

El trabajo se encadena en las siguientes fases, cada una definida como etapa(s) de DVC:

1. **Preprocesamiento** — conversión de xView (GeoJSON) a formato COCO, agrupamiento de las 60
   clases originales en 9 macrocategorías, validación de cajas, partición estratificada 70/10/20
   y recorte de las imágenes en *tiles* de 700×700 px.
2. **Detector base** — entrenamiento de un Faster R-CNN (ResNet-50 + FPN v2) con el subconjunto
   etiquetado (cota inferior, *Faster10*) y con el `train` completo (cota superior).
3. **Inferencia de vocabulario abierto** — Grounding DINO y Rex-Omni sobre las imágenes no
   etiquetadas, con *ensemble* multi-prompt y agregación (*union-find* + voto de Borda).
4. **Selección de pseudoetiquetas** — reglas de selección por fuente y zonas de exclusión.
5. **Reentrenamiento SSOD** — Faster R-CNN con las pseudoetiquetas combinadas (`FT`, `GD`, `RO` y
   sus combinaciones).
6. **Evaluación** — métricas COCO sobre `test` y tabla comparativa final.

---

## Requisitos

**Hardware**

- GNU/Linux con una **GPU NVIDIA** compatible con CUDA 11.8 (imprescindible para entrenamiento e
  inferencia).
- **Disco**: el almacén DVC distribuido ocupa **~62 GB** comprimido; planifica varias decenas de
  GB adicionales para la materialización completa de datos y salidas.

**Software**

- **git** (para clonar) y **Docker** con el **NVIDIA Container Toolkit**.
- Acceso a internet la primera vez: los pesos de Grounding DINO y Rex-Omni se descargan
  automáticamente de HuggingFace.

---

## 1. Obtener el código

```bash
git clone https://github.com/osiiz/ssod-open-vocabulary-xview.git
cd ssod-open-vocabulary-xview
```

---

## 2. Obtener los datos

Hay **dos vías**. La **Vía B (almacén DVC) es la recomendada**: evita la descarga manual de xView
y la reejecución del pipeline. En ambos casos, **obtén los datos antes de crear el entorno**, para
montarlos después.

### Vía A — descarga original de xView

1. Regístrate y descarga el conjunto de entrenamiento de xView (DIUx xView 2018 Detection
   Challenge) en **https://xviewdataset.org/**.
2. Coloca los datos dentro de la carpeta `xView/` (en la raíz del repo) manteniendo la estructura
   original:
   ```
   xView/
   ├── train_images/            # imágenes .tif de entrenamiento (descargadas)
   ├── xView_train.geojson      # anotaciones originales (descargadas)
   ├── xView_classes.json       # ya incluido en este repo (60 clases originales)
   └── xView_macro_classes.json # ya incluido (mapeo a las 9 macrocategorías)
   ```
   La ruta base se define en `params.yaml` (`xview.extracted_data_path: "xView"`). Como `xView/`
   está dentro del repo, se monta automáticamente con el contenedor. Con esta vía hay que
   **ejecutar el pipeline** (Sección 4) para generar las salidas.

### Vía B — almacén DVC precalculado (recomendado)

Se distribuye un **almacén DVC** comprimido (`dvcstore.zip`, ~62 GB) que contiene los datos **y**
las salidas intermedias del pipeline (de cada entrenamiento se guarda solo el mejor *checkpoint*).
Descárgalo de OneDrive:

**https://nubeusc-my.sharepoint.com/:f:/g/personal/lois_fraga_rai_usc_es/IgCr699L1RBuS575_dAzNMYtAZivxxWq-VCx9-yUrzWWLEA?e=iaQW3S**

y descomprímelo en una carpeta local (su contenido es el directorio `tfg_lois_ssod-vocabulario-aberto/`):

```bash
unzip dvcstore.zip -d dvc_store      # crea dvc_store/tfg_lois_ssod-vocabulario-aberto/
#  (si no tienes 'unzip': python -m zipfile -e dvcstore.zip dvc_store)
```

La materialización (`dvc pull`) se hace ya **dentro del entorno** (Sección 3), porque DVC va
instalado en él.

---

## 3. Crear el entorno y materializar los datos

Construye la imagen Docker desde la raíz del repo:

```bash
docker build -f docker/Dockerfile -t tfg_ssod:latest .
```

Lanza el contenedor montando el repo en `/workspace` y, **si usas la Vía B**, el almacén DVC en
`/dvcstore` (la ruta que la configuración de DVC del repo ya espera):

```bash
docker run -it --rm --gpus all --ipc=host \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/dvc_store":/dvcstore \
  tfg_ssod:latest /bin/bash
```

Con la **Vía A** la carpeta `xView/` ya va dentro del repo montado en `/workspace`, así que puedes
omitir el montaje `-v "$(pwd)/dvc_store":/dvcstore`.

Ya dentro del contenedor (el entorno conda se activa solo), con la **Vía B** materializa datos y
salidas:

```bash
dvc pull
```

El `remote` de DVC ya apunta a `/dvcstore/tfg_lois_ssod-vocabulario-aberto`, así que **no hace
falta** `dvc remote modify`. Tras `dvc pull` quedan poblados `xView/`, `results/` y el resto de
salidas tal como se usaron en el trabajo, sin reejecutar nada (se puede traer solo una parte con
`dvc pull <fichero.dvc>` o `dvc pull <nombre_de_stage>`).

---

## 4. Reproducir el pipeline

Con los datos en su sitio (Vía A) o ya materializados (Vía B):

```bash
dvc status         # etapas desactualizadas
dvc repro          # ejecuta lo necesario para ponerlo todo al día
```

- Se necesita **GPU**; la inferencia de **Rex-Omni** (modelo autorregresivo) es la etapa más
  costosa.
- Para fijar la GPU: `CUDA_VISIBLE_DEVICES=0 dvc repro`.
- Para reproducir una etapa concreta: `dvc repro <nombre_de_stage>` (listadas en `dvc.yaml`).
- Todos los parámetros (semillas, umbrales, épocas, etc.) están en `params.yaml`.

> Con la **Vía B** todo está materializado, así que `dvc repro` no reejecuta nada costoso. Las dos
> etapas de figuras (`generate_charts*`) están `frozen` (necesitan los `detection_results.json`
> completos, no distribuidos); `dvc repro` las salta. Las figuras y la tabla comparativa ya están en
> el repo (`docs/charts*/`, `results/ssod/comparison_table.csv`).

---

## 5. Resultados

Las evaluaciones finales sobre `test` (protocolo COCO) quedan en:

```
results/inference_test_ssod_baseline/metrics.json   # cota inferior (Faster10, 10 % etiquetado)
results/inference_test/metrics.json                 # cota superior (train completo)
results/inference_test_ssod_pe/<EXP>/metrics.json   # cada combinación SSOD
```

donde `<EXP>` ∈ {`FT`, `GD`, `RO`, `FT_GD`, `FT_RO`, `FT_GD_RO`, …}. La etapa
`generate_pe_comparison_table` recopila todas las métricas y produce la tabla comparativa y el
gráfico de barras en `docs/`. La métrica de referencia es el **AP50**.

---

## 6. Pruebas

```bash
pytest -v tests
```

---

## 7. Estructura del repositorio

```
src/             código fuente (preprocessing, training, inference, ssod, utils)
scripts/         scripts auxiliares y de ejecución de experimentos
configs/         configuración del modelo y de los prompts (prompt sets)
vendor/Rex-Omni/ wrapper de Rex-Omni usado en la inferencia (código de terceros)
tests/           pruebas con pytest
docker/          Dockerfile y entrypoint
dvc.yaml         definición del pipeline reproducible
params.yaml      parámetros de todas las etapas
environment.yml  dependencias del entorno conda
```

---

## 8. Configuración

- **`params.yaml`** — punto único de configuración, con una sección por fase: `xview`,
  `preprocessing`, `supervised_curve`, `dino`, `rexomni`, `training`, `training_ssod_baseline`,
  `training_ssod_pe`, `pe_policy_ft`, `ssod`, `evaluation`, etc.
- **`configs/models/`** — definición del Faster R-CNN base (ResNet-50 + FPN v2 con *anchors*
  personalizados adaptados al tamaño de los objetos de xView).
- **`configs/prompts/`** — ficheros YAML con los *prompts* de los detectores OV; el principal es
  `single_term_prompts.yaml` (cinco *prompt sets*, un término por clase).

---

## 9. Notas y resolución de problemas

- Los pesos de **Grounding DINO** (`IDEA-Research/grounding-dino-base`) y **Rex-Omni**
  (`IDEA-Research/Rex-Omni`) se descargan de HuggingFace en la primera ejecución; no se incluyen
  en el repositorio (requiere internet y espacio en la caché de HuggingFace).
- **Rex-Omni** se incluye vendorizado en `vendor/Rex-Omni/` y se instala en modo editable
  (necesario para la lógica de extracción de *scores*); lo hace `environment.yml` automáticamente.
- La versión de CUDA que trae PyTorch (11.8) es independiente del *driver* CUDA del sistema y
  compatible con él; no hace falta instalar CUDA aparte.
- Si `unzip` no está instalado, usa `python -m zipfile -e dvcstore.zip <ruta>` (el entorno trae
  Python).
