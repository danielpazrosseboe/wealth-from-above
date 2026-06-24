import table_common as T

TITLE = "Urban\u2013Rural Decomposition of the MIL Gain (West Africa)"
HEADERS = ["Stratum", "n", "Single-patch R\u00b2", "MIL R\u00b2",
           "\u0394 (MIL \u2212 SP)", "95% CI"]
ROWS = [
    ["All",   "3,132", "0.549", "0.596", "+0.047", "[+0.026, +0.072]"],
    ["Rural", "1,888", "0.219", "0.333", "+0.114", "[+0.064, +0.177]"],
    ["Urban", "1,244", "0.190", "0.224", "+0.033", "[\u22120.019, +0.085]"],
]

def table_urban_rural_mil():
    return T.booktabs_png(TITLE, HEADERS, ROWS, ["l", "c", "c", "c", "c", "c"],
                          T.out("table_urban_rural_mil.png"), figw=11.5, aspect=2.7, fontsize=12.5,
                          edges=[0, 0.16, 0.30, 0.49, 0.63, 0.80, 1.0])

if __name__ == "__main__":
    table_urban_rural_mil()
