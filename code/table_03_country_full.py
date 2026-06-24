import analysis_results as A, table_common as T

def table_country_full(s):
    order = sorted(s["country"]["IC"]["fused"], key=lambda c: -s["country"]["IC"]["fused"].get(c, -99))
    rows = [[A.CC2NAME.get(c, c),
             T.fmt(s["country"]["IC"]["ms"].get(c)),  T.fmt(s["country"]["IC"]["nl"].get(c)),  T.fmt(s["country"]["IC"]["fused"].get(c)),
             T.fmt(s["country"]["OOC"]["ms"].get(c)), T.fmt(s["country"]["OOC"]["nl"].get(c)), T.fmt(s["country"]["OOC"]["fused"].get(c))]
            for c in order]
    rows.append(["Pooled", T.fmt(s["pooled"]["IC"]["ms"]), T.fmt(s["pooled"]["IC"]["nl"]), T.fmt(s["pooled"]["IC"]["fused"]),
                 T.fmt(s["pooled"]["OOC"]["ms"]), T.fmt(s["pooled"]["OOC"]["nl"]), T.fmt(s["pooled"]["OOC"]["fused"])])
    return T.academic_table_png("MS / NTL / Fused R\u00b2 - Full Dataset (In-Country and Out-of-Country)",
                                ["Country","IC MS","IC NTL","IC Fused","OOC MS","OOC NTL","OOC Fused"],
                                rows, ["l","c","c","c","c","c","c"], T.out("table_country_full.png"), bold_last=True, figw=13)

if __name__ == "__main__":
    table_country_full(A.r2_summary(A.load_predictions()))
