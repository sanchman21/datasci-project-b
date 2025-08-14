FROM python:3.10-slim

COPY web/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY web/dataset_summaries /app/dataset_summaries
COPY experiments/monocyte/test/figures/clustering /app/experiments/monocyte/test/figures/clustering/
COPY experiments/monocyte/train/monocyte_fold_0_with_TTA/figures/clustering /app/experiments/monocyte/train/monocyte_fold_0_with_TTA/figures/clustering/
COPY experiments/monocyte/train/monocyte_fold_1_with_TTA/figures/clustering /app/experiments/monocyte/train/monocyte_fold_1_with_TTA/figures/clustering/
COPY experiments/monocyte/train/monocyte_fold_2_with_TTA/figures/clustering /app/experiments/monocyte/train/monocyte_fold_2_with_TTA/figures/clustering/
COPY experiments/monocyte/train/monocyte_fold_3_with_TTA/figures/clustering /app/experiments/monocyte/train/monocyte_fold_3_with_TTA/figures/clustering/
COPY experiments/monocyte/train/monocyte_fold_4_with_TTA/figures/clustering /app/experiments/monocyte/train/monocyte_fold_4_with_TTA/figures/clustering/
COPY experiments/neutrophil/test/figures/clustering /app/experiments/neutrophil/test/figures/clustering/
COPY experiments/neutrophil/train/neutrophil_fold_0_with_TTA/figures/clustering /app/experiments/neutrophil/train/neutrophil_fold_0_with_TTA/figures/clustering/
COPY experiments/neutrophil/train/neutrophil_fold_1_with_TTA/figures/clustering /app/experiments/neutrophil/train/neutrophil_fold_1_with_TTA/figures/clustering/
COPY experiments/neutrophil/train/neutrophil_fold_2_with_TTA/figures/clustering /app/experiments/neutrophil/train/neutrophil_fold_2_with_TTA/figures/clustering/
COPY experiments/neutrophil/train/neutrophil_fold_3_with_TTA/figures/clustering /app/experiments/neutrophil/train/neutrophil_fold_3_with_TTA/figures/clustering/
COPY experiments/neutrophil/train/neutrophil_fold_4_with_TTA/figures/clustering /app/experiments/neutrophil/train/neutrophil_fold_4_with_TTA/figures/clustering/
COPY experiments/monocyte_new_normals/test/figures/clustering /app/experiments/monocyte_new_normals/test/figures/clustering/
COPY experiments/monocyte_new_normals/train/monocyte_new_normals_fold_0_with_TTA/figures/clustering /app/experiments/monocyte_new_normals/train/monocyte_new_normals_fold_0_with_TTA/figures/clustering/
COPY experiments/monocyte_new_normals/train/monocyte_new_normals_fold_1_with_TTA/figures/clustering /app/experiments/monocyte_new_normals/train/monocyte_new_normals_fold_1_with_TTA/figures/clustering/
COPY experiments/monocyte_new_normals/train/monocyte_new_normals_fold_2_with_TTA/figures/clustering /app/experiments/monocyte_new_normals/train/monocyte_new_normals_fold_2_with_TTA/figures/clustering/
COPY experiments/monocyte_new_normals/train/monocyte_new_normals_fold_3_with_TTA/figures/clustering /app/experiments/monocyte_new_normals/train/monocyte_new_normals_fold_3_with_TTA/figures/clustering/
COPY experiments/monocyte_new_normals/train/monocyte_new_normals_fold_4_with_TTA/figures/clustering /app/experiments/monocyte_new_normals/train/monocyte_new_normals_fold_4_with_TTA/figures/clustering/
COPY experiments/segmented_monocyte_new_normals/test/figures/clustering /app/experiments/segmented_monocyte_new_normals/test/figures/clustering/
COPY experiments/segmented_monocyte_new_normals/train/segmented_monocyte_new_normals_fold_0_with_TTA/figures/clustering /app/experiments/segmented_monocyte_new_normals/train/segmented_monocyte_new_normals_fold_0_with_TTA/figures/clustering/
COPY experiments/segmented_monocyte_new_normals/train/segmented_monocyte_new_normals_fold_1_with_TTA/figures/clustering /app/experiments/segmented_monocyte_new_normals/train/segmented_monocyte_new_normals_fold_1_with_TTA/figures/clustering/
COPY experiments/segmented_monocyte_new_normals/train/segmented_monocyte_new_normals_fold_2_with_TTA/figures/clustering /app/experiments/segmented_monocyte_new_normals/train/segmented_monocyte_new_normals_fold_2_with_TTA/figures/clustering/
COPY experiments/segmented_monocyte_new_normals/train/segmented_monocyte_new_normals_fold_3_with_TTA/figures/clustering /app/experiments/segmented_monocyte_new_normals/train/segmented_monocyte_new_normals_fold_3_with_TTA/figures/clustering/
COPY experiments/segmented_monocyte_new_normals/train/segmented_monocyte_new_normals_fold_4_with_TTA/figures/clustering /app/experiments/segmented_monocyte_new_normals/train/segmented_monocyte_new_normals_fold_4_with_TTA/figures/clustering/
COPY experiments/segmented_monocyte/test/figures/clustering /app/experiments/segmented_monocyte/test/figures/clustering/
COPY experiments/segmented_monocyte/train/segmented_monocyte_fold_0_with_TTA/figures/clustering /app/experiments/segmented_monocyte/train/segmented_monocyte_fold_0_with_TTA/figures/clustering/
COPY experiments/segmented_monocyte/train/segmented_monocyte_fold_1_with_TTA/figures/clustering /app/experiments/segmented_monocyte/train/segmented_monocyte_fold_1_with_TTA/figures/clustering/
COPY experiments/segmented_monocyte/train/segmented_monocyte_fold_2_with_TTA/figures/clustering /app/experiments/segmented_monocyte/train/segmented_monocyte_fold_2_with_TTA/figures/clustering/
COPY experiments/segmented_monocyte/train/segmented_monocyte_fold_3_with_TTA/figures/clustering /app/experiments/segmented_monocyte/train/segmented_monocyte_fold_3_with_TTA/figures/clustering/
COPY experiments/segmented_monocyte/train/segmented_monocyte_fold_4_with_TTA/figures/clustering /app/experiments/segmented_monocyte/train/segmented_monocyte_fold_4_with_TTA/figures/clustering/

COPY mlruns /app/mlruns/

RUN chmod -R 755 /app/experiments/
RUN chmod -R 755 /app/mlruns/

WORKDIR /app

COPY web/app.py /app/
COPY web/templates/ /app/templates/

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]