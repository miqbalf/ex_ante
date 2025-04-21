import math


def calc_biomass_formula(ttb_formula, wd, dbh, height, text_only=False):

    formula_fix = (
        ttb_formula.replace("TTB=", "")
        .replace("DBH", str(dbh))
        .replace("HEIGHT", str(height))
        .replace("WD", str(wd))
        .replace("exp", "math.exp")
        .replace("EXP", "math.exp")
        .replace("LN(", "math.log(")
        .replace("ln(", "math.log(")
    )

    if "nan" in formula_fix:
        return None

    else:
        formula_fix = formula_fix.replace("^", "**")

        # if you want to check the formula after replace with the i row, you can uncomment this return instead of the following after
        # return formula_fix

        # comment this following if you want to return above

        if text_only == False:
            result_ttb_amount = eval(formula_fix)
            return result_ttb_amount
        else:  # if we want to just check the text only (e.g comparing with treeo cloud ttbFormula!)
            return formula_fix
