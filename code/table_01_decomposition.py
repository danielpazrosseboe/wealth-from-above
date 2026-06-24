import analysis_results as A, table_common as T

def table_decomposition():
    d = A.load_predictions(); b = A.decomposition(d, A._load(A.CONFIG.META_CSV))
    headers = ["Configuration","Stratum","n","MS-only","NTL-only","Fused","NTL gain"]
    rows = [[r["config"], r["scope"], f'{int(r["n"]):,}', T.fmt(r["MS_only"]), T.fmt(r["NL_only"]),
             T.fmt(r["fused"]), T.fmtd(r["NTL_gain"])] for _, r in b.iterrows()]
    return T.academic_table_png("Decomposition of Predictive Performance (MS-only, NTL-only, Fused R\u00b2)",
        headers, rows, ["l","l","c","c","c","c","c"], T.out("table_decomposition.png"), bold_last=False, figw=12)

if __name__ == "__main__":
    table_decomposition()
