import table_common as T

TITLE = "Benchmarking Against the Relative Wealth Index, by Temporal Match"
HEADERS = ["Model / subset", "n", "Model R\u00b2", "RWI R\u00b2",
           "\u0394 (Model \u2212 RWI)", "95% CI"]
ROWS = [
    ["In-country: All matched",                  "16,279", "0.682", "0.490", "+0.192", "[+0.182, +0.202]"],
    ["In-country: Contemporaneous (2019\u20132021)", "4,678",  "0.686", "0.491", "+0.196", "[+0.177, +0.215]"],
    ["Out-of-country: All matched",              "16,279", "0.639", "0.490", "+0.148", "[+0.139, +0.157]"],
    ["Out-of-country: Contemporaneous (2019\u20132021)", "4,678", "0.663", "0.491", "+0.172", "[+0.156, +0.189]"],
]

def table_temporal_robustness():
    return T.booktabs_png(TITLE, HEADERS, ROWS, ["l", "c", "c", "c", "c", "c"],
                          T.out("table_temporal_robustness.png"), figw=12.5, aspect=2.55, fontsize=12,
                          edges=[0, 0.40, 0.50, 0.61, 0.72, 0.855, 1.0])

if __name__ == "__main__":
    table_temporal_robustness()
