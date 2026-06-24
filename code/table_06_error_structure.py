import analysis_results as A, table_common as T

def table_bias():
    c, _ = A.bias_analysis(A.load_predictions())
    headers = ["Configuration","Slope\n(pred on true)","Resid. Q1\n(poorest)","Resid. Q5\n(richest)","Resid. SD"]
    rows = [[r["config"], T.fmt(r["slope_pred_on_true"]), T.fmtd(r["mean_resid_Q1"]),
             T.fmtd(r["mean_resid_Q5"]), T.fmt(r["resid_SD"])] for _, r in c.iterrows()]
    return T.academic_table_png("Error-Structure Summary for the Fused Model", headers, rows,
        ["l","c","c","c","c"], T.out("table_bias.png"), bold_last=False, figw=11)

if __name__ == "__main__":
    table_bias()
