import analysis_results as A, table_common as T

def table_wa_mil(s):
    if "MIL" not in s["pooled"]: print("[SKIP] MIL absent"); return None
    order = sorted(A.WA, key=lambda c: -s["country"]["MIL"]["fused"].get(c, -99))
    rows = [[A.CC2NAME.get(c, c),
             T.fmt(s["country"]["WA"]["ms"].get(c)),  T.fmt(s["country"]["WA"]["nl"].get(c)),  T.fmt(s["country"]["WA"]["fused"].get(c)),
             T.fmt(s["country"]["MIL"]["ms"].get(c)), T.fmt(s["country"]["MIL"]["nl"].get(c)), T.fmt(s["country"]["MIL"]["fused"].get(c))]
            for c in order]
    rows.append(["Pooled", T.fmt(s["pooled"]["WA"]["ms"]), T.fmt(s["pooled"]["WA"]["nl"]), T.fmt(s["pooled"]["WA"]["fused"]),
                 T.fmt(s["pooled"]["MIL"]["ms"]), T.fmt(s["pooled"]["MIL"]["nl"]), T.fmt(s["pooled"]["MIL"]["fused"])])
    return T.academic_table_png("MS / NTL / Fused R\u00b2 - West Africa: Single-Patch vs MIL 3\u00d73",
                                ["Country","Single MS","Single NTL","Single Fused","MIL MS","MIL NTL","MIL Fused"],
                                rows, ["l","c","c","c","c","c","c"], T.out("table_wa_mil.png"), bold_last=True, figw=12)

if __name__ == "__main__":
    table_wa_mil(A.r2_summary(A.load_predictions()))
