import analysis_results as A, table_common as T

def table_district():
    d = A.load_predictions()
    try: cl, di = A.district_aggregation(d)
    except Exception as e: print(f"[SKIP] district table - inputs missing: {e}"); return None
    rows = [[name, T.fmt(cl[k]), T.fmt(di[k]), T.fmtd(di[k] - cl[k])]
            for k, name in [("IC","Full-IC"),("OOC","Full-OOC"),("WA","WA-SP"),("MIL","WA-MIL")] if k in di]
    return T.academic_table_png("Pooled Fused R\u00b2 - Cluster vs District Level (ADM2)",
        ["Configuration","Cluster R\u00b2","District R\u00b2","\u0394 (District \u2212 Cluster)"],
        rows, ["l","c","c","c"], T.out("table_district.png"), bold_last=False, figw=11)

if __name__ == "__main__":
    table_district()
