import table_common as T

PC1_LOADINGS = [("TV","0.44"),("Electricity","0.43"),("Refrigerator","0.37"),("Toilet","0.36"),
                ("Floor","0.35"),("Water","0.33"),("Car","0.24"),("Radio","0.21"),
                ("Motorbike","0.10"),("Phone","0.09"),("Rooms per person","0.09")]
PC1_VAR_EXPLAINED = "30.9%"

def table_a1_loadings():
    rows = [[name, val] for name, val in PC1_LOADINGS]
    return T.academic_table_png(f"Table A1. PC1 wealth-index loadings (PC1 = {PC1_VAR_EXPLAINED} of variance)",
                                ["Asset variable", "PC1 loading"], rows, ["l","c"],
                                T.out("table_A1_pc1_loadings.png"), bold_last=False, figw=6, label_frac=0.55)

if __name__ == "__main__":
    table_a1_loadings()
