import pandas as pd, analysis_results as A, table_common as T

def table_perfold():
    a = A.per_fold_spread(A.load_predictions())
    headers = ["Configuration","Pooled R\u00b2","Folds","Min","Max","Range","SD"]
    rows = [[r["config"], T.fmt(r["pooled_R2"]), int(r["n_folds"]) if pd.notna(r.get("n_folds")) else "\u2014",
             T.fmt(r["fold_min"]), T.fmt(r["fold_max"]), T.fmt(r["fold_range"]), T.fmt(r["fold_SD"])]
            for _, r in a.iterrows()]
    return T.academic_table_png("Spread of the Fused R\u00b2 Across Cross-Validation Folds", headers, rows,
        ["l","c","c","c","c","c","c"], T.out("table_perfold_spread.png"), bold_last=False, figw=11)

if __name__ == "__main__":
    table_perfold()
