import numpy as np, analysis_results as A, table_common as T

def table_summary(s):
    if "MIL" not in s["pooled"]: print("[SKIP] MIL absent"); return None
    order = sorted(A.WA, key=lambda c: -s["country"]["MIL"]["fused"].get(c, -99))
    rows = []
    for c in order:
        wa_v, mil_v = s["country"]["WA"]["fused"].get(c, np.nan), s["country"]["MIL"]["fused"].get(c, np.nan)
        delta = mil_v - wa_v if np.isfinite(mil_v) and np.isfinite(wa_v) else np.nan
        rows.append([A.CC2NAME.get(c, c), T.fmt(s["country"]["IC"]["fused"].get(c)), T.fmt(s["country"]["OOC"]["fused"].get(c)),
                     T.fmt(wa_v), T.fmt(mil_v), T.fmtd(delta)])
    rows.append(["POOLED", T.fmt(s["pooled"]["IC"]["fused"]), T.fmt(s["pooled"]["OOC"]["fused"]),
                 T.fmt(s["pooled"]["WA"]["fused"]), T.fmt(s["pooled"]["MIL"]["fused"]),
                 T.fmtd(s["pooled"]["MIL"]["fused"] - s["pooled"]["WA"]["fused"])])
    return T.styled_table_png("Fused R\u00b2 by country \u2014 all models",
                              ["Country","In-Country","Out-of-Country","Single-Patch","MIL 3x3","\u0394 MIL"],
                              rows, [1,2,3,4], T.out("table_summary.png"), delta_col=5, figsize=(13,7), fontsize=11.5)

if __name__ == "__main__":
    table_summary(A.r2_summary(A.load_predictions()))
