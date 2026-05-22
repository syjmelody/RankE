import os
import json
import argparse
import numpy as np

# Import evaluation directly from the local HPSv2 package
# This assumes you already ran `pip install -e .` for the local HPSv2 package,
# or that your PYTHONPATH already includes the HPSv2 repository root.
try:
    from hpsv2 import evaluation as hps_eval
except ImportError:
    print("[Error] Failed to import hpsv2. Please make sure the local HPSv2 repository is in your PYTHONPATH.")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="Offline custom HPSv2 Evaluation")
    parser.add_argument("--image_path", type=str, required=True, help="Root directory of generated images (containing subfolders such as anime and photo)")
    parser.add_argument("--hpdv2_path", type=str, required=True, help="Local HPDv2 dataset directory (should contain benchmark/xxx.json)")
    parser.add_argument("--hps_model_path", type=str, required=True, help="Absolute path to the local HPS_v2.1_compressed.pt checkpoint")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    # HPSv2 evaluation.evaluate(mode="benchmark") requires a data_path argument
    # That data_path must point to the directory that contains anime.json and related files
    # In practice this is typically the HPDv2/benchmark directory
    benchmark_data_path = os.path.join(args.hpdv2_path, "benchmark")
    if not os.path.exists(benchmark_data_path):
        print(f"[Error] Benchmark data path not found: {benchmark_data_path}")
        exit(1)

    print(f">>> Starting offline HPSv2 evaluation")
    print(f"    - Model path: {args.hps_model_path}")
    print(f"    - Data path: {benchmark_data_path}")
    print(f"    - Image path: {args.image_path}")

    # Call the local HPSv2 evaluate function.
    # As long as checkpoint_path and data_path are given explicitly, it will not trigger huggingface_hub_download.
    # Note: the upstream evaluate_benchmark prints scores instead of returning them, so this script captures the relevant outputs explicitly.
    # For a more reliable score.txt output, this script reproduces the minimal benchmark scoring logic locally.

    # ==========================
    # Re-implemented offline scoring core
    # ==========================
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    
    # Initialize the model
    hps_eval.initialize_model()
    model = hps_eval.model_dict['model']
    preprocess_val = hps_eval.model_dict['preprocess_val']

    print('Loading model ...')
    checkpoint = __import__('torch').load(args.hps_model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = hps_eval.get_tokenizer(hps_eval.model_name)
    model = model.to(device)
    model.eval()
    print('Loading model successfully!')

    # Score each style category
    style_list = [d for d in os.listdir(args.image_path) if os.path.isdir(os.path.join(args.image_path, d))]
    
    final_scores = {}
    all_scores = []

    for style in style_list:
        image_folder = os.path.join(args.image_path, style)
        
        # Support both HPDv2 JSON naming conventions
        meta_file = os.path.join(benchmark_data_path, f'{style}.json')
        if not os.path.exists(meta_file):
            meta_file = os.path.join(benchmark_data_path, f'benchmark_{style}.json')
            if not os.path.exists(meta_file):
                print(f"[Warning] Meta file not found for style {style}. Skipping.")
                continue

        print(f"Evaluating style: {style} ...")
        dataset = hps_eval.BenchmarkDataset(meta_file, image_folder, preprocess_val, tokenizer)
        dataloader = hps_eval.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=hps_eval.collate_eval, num_workers=4)

        style_scores = []
        with __import__('torch').no_grad():
            for batch in dataloader:
                images, texts = batch
                images = images.to(device=device, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with __import__('torch').cuda.amp.autocast():
                    outputs = model(images, texts)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    # GCPO and the upstream default do not multiply scores by 100, so this keeps the native HPS range (for example 0.25-0.30).
                    logits_per_image = image_features @ text_features.T
                
                style_scores.extend(__import__('torch').diagonal(logits_per_image).cpu().tolist())
        
        if len(style_scores) > 0:
            avg_style_score = np.mean(style_scores)
            final_scores[style] = avg_style_score
            all_scores.extend(style_scores)
            print(f"  --> {style}: {avg_style_score:.4f}")

    # ==========================
    # Write score.txt
    # ==========================
    score_file = os.path.join(args.image_path, "score.txt")
    with open(score_file, "w") as f:
        f.write("HPS v2.1 Scores:\n")
        for style, score in final_scores.items():
            f.write(f"{style}: {score:.4f}\n")
        
        if len(all_scores) > 0:
            avg_score = np.mean(all_scores)
            f.write(f"Average: {avg_score:.4f}\n")
            print(f"\nOverall Average: {avg_score:.4f}")
    
    print(f"\n>>> [Done] Scores have been saved to {score_file}")

if __name__ == "__main__":
    main()