import os
import json
from flask import Flask, render_template, request, abort, send_file, redirect, url_for
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import mlflow
from mlflow.server import app as mlflow_app

app = Flask(__name__)
mlflow.set_tracking_uri("file:///app/mlruns")
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    "/mlflow": mlflow_app
})

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), ".", "experiments")
DATA_TYPES = ["monocyte", "neutrophil", "monocyte_new_normals", "segmented_monocyte", "segmented_monocyte_new_normals"]
COMPARABLE_DATA_TYPES = ["monocyte", "monocyte_new_normals"]

def get_clustering_file(data_type, fold):
    if data_type not in DATA_TYPES:
        print(f"Unsupported data type: {data_type}")
        return None
    if fold not in ["Test", "0", "1", "2", "3", "4"]:
        print(f"Invalid fold: {fold}")
        return None
    if fold == "Test":
        html_dir = os.path.join(EXPERIMENTS_DIR, data_type, "test", "figures", "clustering")
        html_path = os.path.join(html_dir, "clustering_final.html")
        fold_display = "final"
    else:
        html_dir = os.path.join(EXPERIMENTS_DIR, data_type, "train", f"{data_type}_fold_{fold}_with_TTA", "figures", "clustering")
        html_path = os.path.join(html_dir, f"clustering_fold_{fold}.html")
        fold_display = fold
    if not os.path.exists(html_path):
        print(f"HTML file not found at {html_path}")
        return None
    
    cluster_metadata_path = os.path.join(html_dir, "cluster_metadata.json")
    misclassified_images_path = os.path.join(html_dir, "misclassified_images.json")
    cluster_metadata = {}
    misclassified_images = {}
    
    if os.path.exists(cluster_metadata_path):
        with open(cluster_metadata_path, 'r') as f:
            cluster_metadata = json.load(f)
    else:
        print(f"Cluster metadata not found at {cluster_metadata_path}")
        
    if os.path.exists(misclassified_images_path):
        with open(misclassified_images_path, 'r') as f:
            misclassified_images = json.load(f)
    else:
        print(f"Misclassified images metadata not found at {misclassified_images_path}")
        
    for cluster in cluster_metadata:
        if cluster not in ["patient_to_cluster", "clustering_metrics", "feature_importance", "shap_plot"]:
            median_image = cluster_metadata[cluster].get("median_image")
            if median_image and not os.path.exists(os.path.join(os.path.dirname(__file__), median_image)):
                print(f"Warning: Median image not found at {os.path.join(os.path.dirname(__file__), median_image)} for cluster {cluster}")
                cluster_metadata[cluster]["median_image"] = None
            median_gradcam = cluster_metadata[cluster].get("median_gradcam")
            if median_gradcam and not os.path.exists(os.path.join(os.path.dirname(__file__), median_gradcam)):
                print(f"Warning: Median Grad-CAM not found at {os.path.join(os.path.dirname(__file__), median_gradcam)} for cluster {cluster}")
                cluster_metadata[cluster]["median_gradcam"] = None
            top_k_images = cluster_metadata[cluster].get("top_k_images", [])
            valid_top_k = []
            for img_path in top_k_images:
                if os.path.exists(os.path.join(os.path.dirname(__file__), img_path)):
                    valid_top_k.append(img_path)
                else:
                    print(f"Warning: Top-K image not found at {os.path.join(os.path.dirname(__file__), img_path)} for cluster {cluster}")
            cluster_metadata[cluster]["top_k_images"] = valid_top_k
            
    for patient, details in misclassified_images.items():
        for img in details["images"]:
            gradcam_path = img.get("gradcam_path")
            if gradcam_path and not os.path.exists(os.path.join(os.path.dirname(__file__), gradcam_path)):
                print(f"Warning: Grad-CAM file not found at {os.path.join(os.path.dirname(__file__), gradcam_path)} for patient {patient}")
                img["gradcam_path"] = None
            img_path = img.get("path")
            if img_path and not os.path.exists(os.path.join(os.path.dirname(__file__), img_path)):
                print(f"Warning: Misclassified image not found at {os.path.join(os.path.dirname(__file__), img_path)} for patient {patient}")
                img["path"] = None
                
    return {
        "data_type": data_type,
        "fold": fold_display,
        "html_path": html_path,
        "cluster_metadata": cluster_metadata,
        "misclassified_images": misclassified_images,
        "relative_html_path": os.path.relpath(html_path, start=os.path.dirname(__file__))
    }
    
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data_type = request.form.get('data_type')
        fold = request.form.get('fold')
        if data_type and fold:
            return redirect(url_for('clustering', data_type=data_type, fold=fold))
    return render_template('index.html', data_types=DATA_TYPES, folds=["Test", "0", "1", "2", "3", "4"])

@app.route('/clustering/<data_type>/<fold>')
def clustering(data_type, fold):
    clustering_file = get_clustering_file(data_type, fold)
    if not clustering_file:
        abort(404)
    return render_template('clustering.html', clustering_file=clustering_file)

@app.route('/compare_clusters', methods=['GET', 'POST'])
def compare_clusters():
    if request.method == 'POST':
        normal_data_type = request.form.get('normal_data_type')
        fold = request.form.get('fold')
        if normal_data_type in COMPARABLE_DATA_TYPES and fold in ["Test", "0", "1", "2", "3", "4"]:
            segmented_data_type = "segmented_" + normal_data_type
            return redirect(url_for('compare_clusters_results', normal_data_type=normal_data_type, segmented_data_type=segmented_data_type, fold=fold))
    return render_template('compare_clusters.html', data_types=COMPARABLE_DATA_TYPES, folds=["Test", "0", "1", "2", "3", "4"])

@app.route('/compare_clusters_results/<normal_data_type>/<segmented_data_type>/<fold>')
def compare_clusters_results(normal_data_type, segmented_data_type, fold):
    normal_clustering_file = get_clustering_file(normal_data_type, fold)
    segmented_clustering_file = get_clustering_file(segmented_data_type, fold)
    if not normal_clustering_file or not segmented_clustering_file:
        abort(404, description="One or both clustering files are unavailable.")
    return render_template('compare_clusters_results.html', normal_clustering_file=normal_clustering_file, segmented_clustering_file=segmented_clustering_file)

@app.route('/file/<path:file_path>')
def serve_file(file_path):
    file_path = os.path.normpath(file_path)
    full_path = os.path.join(os.path.dirname(__file__), file_path)
    print(f"Requested file path: {file_path}")
    print(f"Full path on server: {full_path}")
    print(f"File exists: {os.path.exists(full_path)}")
    if os.path.exists(full_path) and os.path.isfile(full_path):
        print(f"Serving file: {full_path}")
        return send_file(full_path)
    else:
        print(f"File not found: {full_path}")
        abort(404)
        
@app.route('/dataset_summary.html')
def dataset_summary():
    return render_template('dataset_summary.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)