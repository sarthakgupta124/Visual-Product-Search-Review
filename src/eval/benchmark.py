import numpy as np
import pandas as pd
import json
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.logger import get_logger
from src.exception import CustomException
from src.utils import get_custom_paths
import sys

logger = get_logger(__name__)


def evaluate_retrieval(search_engine, eval_df, k_values=(1, 5, 10)):
    """
    Evaluate retrieval performance using Recall@k metrics.
    
    Args:
        search_engine: SearchEngine instance
        eval_df: Evaluation DataFrame with 'review_text' and 'asin' columns
        k_values: Tuple of k values for recall calculation
    
    Returns:
        Dictionary with Recall@k scores
    """
    try:
        recalls = {k: [] for k in k_values}
        total_queries = len(eval_df)
        
        logger.info(f"Starting retrieval evaluation on {total_queries} queries...")
        
        for idx, (_, row) in enumerate(eval_df.iterrows()):
            if idx % 100 == 0:
                logger.info(f"  Progress: {idx}/{total_queries}")
            
            results = search_engine.search_by_text(row["review_text"], k=max(k_values))
            
            # Check if 'asin' column exists in retrieved reviews
            if "asin" in results["top_reviews"].columns:
                retrieved_asins = results["top_reviews"]["asin"].tolist()
                for k in k_values:
                    recalls[k].append(row["asin"] in retrieved_asins[:k])
            else:
                logger.warning("ASIN column not found in retrieved reviews. Skipping retrieval evaluation.")
                for k in k_values:
                    recalls[k].append(False)
        
        result = {f"recall@{k}": float(np.mean(v)) for k, v in recalls.items()}
        logger.info(f"Retrieval evaluation completed: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error in evaluate_retrieval: {str(e)}")
        raise CustomException(e, sys)


def evaluate_rating(search_engine, eval_df, k=5, min_sim=0.7):
    """
    Evaluate rating prediction performance.
    
    Args:
        search_engine: SearchEngine instance
        eval_df: Evaluation DataFrame with 'review_text' and 'rating' columns
        k: Number of similar reviews to retrieve
        min_sim: Minimum similarity threshold for rating calculation
    
    Returns:
        Dictionary with MAE and RMSE metrics
    """
    try:
        preds, truths = [], []
        total_queries = len(eval_df)
        
        logger.info(f"Starting rating evaluation on {total_queries} queries (k={k}, min_sim={min_sim})...")
        
        for idx, (_, row) in enumerate(eval_df.iterrows()):
            if idx % 100 == 0:
                logger.info(f"  Progress: {idx}/{total_queries}")
            
            results = search_engine.search_by_text(row["review_text"], k=k)
            pred = search_engine.compute_final_rating(
                results["top_reviews"], results["similarities"], min_sim=min_sim
            )
            preds.append(pred)
            truths.append(row["rating"])
        
        mae = mean_absolute_error(truths, preds)
        rmse = mean_squared_error(truths, preds, squared=False)
        
        result = {
            "mae": float(mae),
            "rmse": float(rmse),
            "samples": len(preds)
        }
        logger.info(f"Rating evaluation completed: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error in evaluate_rating: {str(e)}")
        raise CustomException(e, sys)


def trivial_baseline_mae(eval_df, train_df):
    """
    Compute baseline MAE by predicting the global average rating.
    
    This serves as a sanity check: any model should outperform this trivial baseline.
    
    Args:
        eval_df: Evaluation DataFrame with 'rating' column
        train_df: Training DataFrame with 'rating' column
    
    Returns:
        MAE of the baseline (float)
    """
    try:
        global_avg = train_df["rating"].mean()
        preds = [global_avg] * len(eval_df)
        baseline_mae = mean_absolute_error(eval_df["rating"], preds)
        
        logger.info(f"Baseline MAE (predicting global avg {global_avg:.2f}): {baseline_mae:.4f}")
        return float(baseline_mae)
    
    except Exception as e:
        logger.error(f"Error in trivial_baseline_mae: {str(e)}")
        raise CustomException(e, sys)


def run_evaluation(eval_split=0.2, save_results=True):
    """
    Run comprehensive evaluation of the search and rating system.
    
    Args:
        eval_split: Fraction of data to use for evaluation
        save_results: Whether to save results to JSON file
    
    Returns:
        Dictionary with all evaluation metrics
    """
    try:
        from src.components.inference.search_engine import SearchEngine
        paths = get_custom_paths()

        train_df = pd.read_csv(paths['ingestion_data_path'])
        train_df = train_df.dropna(subset=['review_text', 'rating']).reset_index(drop=True)

        eval_df = pd.read_csv(paths['eval_data_path'])
        eval_df = eval_df.dropna(subset=['review_text', 'rating']).reset_index(drop=True)

        logger.info(f"Indexed/train set: {len(train_df)} | True held-out eval set: {len(eval_df)}")

        search_engine = SearchEngine()
        rating_results = evaluate_rating(search_engine, eval_df, k=5, min_sim=0.7)
        baseline_mae = trivial_baseline_mae(eval_df, train_df)
        
        # Compile results
        results = {
            "eval_config": {
                "eval_split": eval_split,
                "eval_samples": len(eval_df),
                "train_samples": len(train_df),
            },
            "rating_prediction": rating_results,
            "baseline": {
                "trivial_mae": baseline_mae
            },
            "improvement": {
                "vs_baseline": float((baseline_mae - rating_results['mae']) / baseline_mae * 100)
            }
        }
        
        logger.info("-" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info(json.dumps(results, indent=2))
        
        # Save results
        if save_results:
            output_dir = os.path.join(paths['PROJECT_ROOT'], 'eval_results')
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, 'benchmark_results.json')
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to: {output_path}")
        
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETED")
        logger.info("=" * 60)
        
        return results
    
    except Exception as e:
        logger.error(f"Error in run_evaluation: {str(e)}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        results = run_evaluation(eval_split=0.2, save_results=True)
        print("\n✅ Evaluation completed successfully!")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        sys.exit(1)