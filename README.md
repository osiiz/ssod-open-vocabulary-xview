# Detección semi-supervisada con detectores de vocabulario aberto sobre xView

**Galego** · [English](README.en.md) · [Español](README.es.md)

Código do Traballo de Fin de Grao sobre **detección de obxectos semi-supervisada (SSOD)** en
imaxe satelital (conxunto de datos **xView**), que integra os detectores de vocabulario aberto
**Grounding DINO** e **Rex-Omni** conxelados como fonte complementaria de pseudo-etiquetas sobre
un detector base **Faster R-CNN**.

Todo o fluxo está definido como un **pipeline reproducible con [DVC](https://dvc.org)** en
`dvc.yaml`, parametrizado en `params.yaml`. Este repositorio contén o **código** e as **métricas de resultados** (ficheiros JSON pequenos); os
datos, as saídas intermedias voluminosas e os pesos dos modelos non se inclúen, e obtéñense como se
describe na [Sección 2](#2-obter-os-datos).

---

## Visión xeral do pipeline

O traballo encadéase nas seguintes fases, cada unha definida como etapa(s) de DVC:

1. **Preprocesamento** — conversión de xView (GeoJSON) a formato COCO, agrupamento das 60 clases
   orixinais en 9 macro-categorías, validación de caixas, partición estratificada 70/10/20 e
   recorte das imaxes en *tiles* de 700×700 px.
2. **Detector base** — adestramento dun Faster R-CNN (ResNet-50 + FPN v2) co subconxunto
   etiquetado (cota inferior, *Faster10*) e co `train` completo (cota superior).
3. **Inferencia de vocabulario aberto** — Grounding DINO e Rex-Omni sobre as imaxes non
   etiquetadas, con *ensemble* multi-prompt e agregación (*union-find* + voto de Borda).
4. **Selección de pseudo-etiquetas** — regras de selección por fonte e zonas de exclusión.
5. **Readestramento SSOD** — Faster R-CNN coas pseudo-etiquetas combinadas (`FT`, `GD`, `RO` e as
   súas combinacións).
6. **Avaliación** — métricas COCO sobre `test` e táboa comparativa final.

---

## Requisitos

**Hardware**

- GNU/Linux cunha **GPU NVIDIA** compatible con CUDA 11.8 (imprescindible para adestramento e
  inferencia).
- **Disco**: o almacén DVC distribuído ocupa **~62 GB** comprimido; planifica varias decenas de
  GB adicionais para a materialización completa de datos e saídas.

**Software**

- **git** (para clonar) e **Docker** co **NVIDIA Container Toolkit**.
- Acceso a internet a primeira vez: os pesos de Grounding DINO e Rex-Omni descárganse
  automaticamente de HuggingFace.

---

## 1. Obter o código

```bash
git clone https://github.com/osiiz/ssod-open-vocabulary-xview.git
cd ssod-open-vocabulary-xview
```

---

## 2. Obter os datos

Hai **dúas vías**. A **Vía B (almacén DVC) é a recomendada**: evita a descarga manual de xView e
a reexecución do pipeline. En ambos casos, **obtén os datos antes de crear o contorno**, para
montalos despois.

### Vía A — descarga orixinal de xView

1. Rexístrate e descarga o conxunto de adestramento de xView (DIUx xView 2018 Detection
   Challenge) en **https://xviewdataset.org/**.
2. Coloca os datos dentro da carpeta `xView/` (na raíz do repo) mantendo a estrutura orixinal:
   ```
   xView/
   ├── train_images/            # imaxes .tif de adestramento (descargadas)
   ├── xView_train.geojson      # anotacións orixinais (descargadas)
   ├── xView_classes.json       # xa incluído neste repo (60 clases orixinais)
   └── xView_macro_classes.json # xa incluído (mapeo ás 9 macro-categorías)
   ```
   A ruta base defínese en `params.yaml` (`xview.extracted_data_path: "xView"`). Como `xView/`
   está dentro do repo, montarase só co contedor. Con esta vía cómpre **executar o pipeline**
   (Sección 4) para xerar as saídas.

### Vía B — almacén DVC precalculado (recomendado)

Distribúese un **almacén DVC** comprimido (`dvcstore.zip`, ~62 GB) que contén os datos **e** as
saídas intermedias do pipeline (de cada adestramento gárdase só o mellor *checkpoint*).
Descárgao de OneDrive:

**https://nubeusc-my.sharepoint.com/:f:/g/personal/lois_fraga_rai_usc_es/IgCr699L1RBuS575_dAzNMYtAZivxxWq-VCx9-yUrzWWLEA?e=iaQW3S**

e descomprímeo nunha carpeta local (o seu contido é o directorio `tfg_lois_ssod-vocabulario-aberto/`):

```bash
unzip dvcstore.zip -d dvc_store      # crea dvc_store/tfg_lois_ssod-vocabulario-aberto/
#  (se non tes 'unzip': python -m zipfile -e dvcstore.zip dvc_store)
```

A materialización (`dvc pull`) faise xa **dentro do contorno** (Sección 3), porque DVC vai
instalado nel.

---

## 3. Crear o contorno e materializar os datos

Constrúe a imaxe Docker desde a raíz do repo:

```bash
docker build -f docker/Dockerfile -t tfg_ssod:latest .
```

Lanza o contedor montando o repo en `/workspace` e, **se usas a Vía B**, o almacén DVC en
`/dvcstore` (a ruta que a configuración de DVC do repo xa espera):

```bash
docker run -it --rm --gpus all --ipc=host \
  -v "$(pwd)":/workspace \
  -v "$(pwd)/dvc_store":/dvcstore \
  tfg_ssod:latest /bin/bash
```

Coa **Vía A** a carpeta `xView/` xa vai dentro do repo montado en `/workspace`, polo que podes
omitir a montaxe `-v "$(pwd)/dvc_store":/dvcstore`.

Xa dentro do contedor (o contorno conda actívase só), coa **Vía B** materializa datos e saídas:

```bash
dvc pull
```

O `remote` de DVC xa apunta a `/dvcstore/tfg_lois_ssod-vocabulario-aberto`, polo que **non fai
falta** `dvc remote modify`. Tras `dvc pull` quedan poboados `xView/`, `results/` e o resto de
saídas tal e como se usaron no traballo, sen reexecutar nada (pódese traer só unha parte con
`dvc pull <ficheiro.dvc>` ou `dvc pull <nome_de_stage>`).

---

## 4. Reproducir o pipeline

Cos datos no sitio (Vía A) ou xa materializados (Vía B):

```bash
dvc status         # etapas desactualizadas
dvc repro          # executa o necesario para poñer todo ao día
```

- Cómpre **GPU**; a inferencia de **Rex-Omni** (modelo autoregresivo) é a etapa máis custosa.
- Para fixar a GPU: `CUDA_VISIBLE_DEVICES=0 dvc repro`.
- Para reproducir unha etapa concreta: `dvc repro <nome_de_stage>` (lista en `dvc.yaml`).
- Todos os parámetros (sementes, limiares, épocas, etc.) están en `params.yaml`.

> Coa **Vía B** todo está materializado, polo que `dvc repro` non reexecuta nada custoso. As dúas
> etapas de figuras (`generate_charts*`) están `frozen` (precisan os `detection_results.json`
> completos, non distribuídos); `dvc repro` sáltaas. As figuras e a táboa comparativa xa están no
> repo (`docs/charts*/`, `results/ssod/comparison_table.csv`).

---

## 5. Resultados

As avaliacións finais sobre `test` (protocolo COCO) quedan en:

```
results/inference_test_ssod_baseline/metrics.json   # cota inferior (Faster10, 10 % etiquetado)
results/inference_test/metrics.json                 # cota superior (train completo)
results/inference_test_ssod_pe/<EXP>/metrics.json   # cada combinación SSOD
```

onde `<EXP>` ∈ {`FT`, `GD`, `RO`, `FT_GD`, `FT_RO`, `FT_GD_RO`, …}. A etapa
`generate_pe_comparison_table` recompila todas as métricas e produce a táboa comparativa e a
gráfica de barras en `docs/`. A métrica de referencia é o **AP50**.

---

## 6. Probas

```bash
pytest -v tests
```

---

## 7. Estrutura do repositorio

```
src/             código fonte (preprocessing, training, inference, ssod, utils)
scripts/         scripts auxiliares e de execución de experimentos
configs/         configuración do modelo e dos prompts (prompt sets)
vendor/Rex-Omni/ wrapper de Rex-Omni empregado na inferencia (código de terceiros)
tests/           probas con pytest
docker/          Dockerfile e entrypoint
dvc.yaml         definición do pipeline reproducible
params.yaml      parámetros de todas as etapas
environment.yml  dependencias da contorna conda
```

---

## 8. Configuración

- **`params.yaml`** — punto único de configuración, con seccións por fase: `xview`,
  `preprocessing`, `supervised_curve`, `dino`, `rexomni`, `training`, `training_ssod_baseline`,
  `training_ssod_pe`, `pe_policy_ft`, `ssod`, `evaluation`, etc.
- **`configs/models/`** — definición do Faster R-CNN base (ResNet-50 + FPN v2 con *anchors*
  personalizadas adaptadas ao tamaño dos obxectos de xView).
- **`configs/prompts/`** — ficheiros YAML cos *prompts* dos detectores OV; o principal é
  `single_term_prompts.yaml` (cinco *prompt sets*, un termo por clase).

---

## 9. Notas e resolución de problemas

- Os pesos de **Grounding DINO** (`IDEA-Research/grounding-dino-base`) e **Rex-Omni**
  (`IDEA-Research/Rex-Omni`) descárganse de HuggingFace na primeira execución; non se inclúen no
  repositorio (require internet e espazo na caché de HuggingFace).
- **Rex-Omni** inclúese vendorizado en `vendor/Rex-Omni/` e instálase en modo editable (necesario
  para a lóxica de extracción de *scores*); faino `environment.yml` automaticamente.
- A versión de CUDA que trae PyTorch (11.8) é independente do *driver* CUDA do sistema e
  compatible con el; non fai falta instalar CUDA á parte.
- Se `unzip` non está instalado, usa `python -m zipfile -e dvcstore.zip <ruta>` (a contorna trae
  Python).
