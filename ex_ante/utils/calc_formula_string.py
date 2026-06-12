import ast
import math
import re


ALLOWED_FORMULA_FUNCTIONS = {
    "abs": abs,
    "exp": math.exp,
    "ln": math.log,
    "log": math.log,
    "pow": pow,
    "sqrt": math.sqrt,
}
ALLOWED_FORMULA_NAMES = set(ALLOWED_FORMULA_FUNCTIONS) | {"e", "pi"}


def calc_biomass_formula(ttb_formula, wd, dbh, height, text_only=False):
    formula_fix = str(ttb_formula or "").strip().strip('"').strip("'")
    if formula_fix.upper().startswith("TTB="):
        formula_fix = formula_fix[4:].strip()

    formula_fix = (
        formula_fix.replace("DBH", str(dbh))
        .replace("HEIGHT", str(height))
        .replace("WD", str(wd))
        .replace("Math.", "")
        .replace("math.", "")
        .replace("EXP", "exp")
        .replace("LN(", "log(")
        .replace("ln(", "log(")
    )
    formula_fix = re.sub(r"\s+", " ", formula_fix).strip()

    if "nan" in formula_fix:
        return None

    else:
        formula_fix = formula_fix.replace("^", "**")

        # if you want to check the formula after replace with the i row, you can uncomment this return instead of the following after
        # return formula_fix

        # comment this following if you want to return above

        if text_only == False:
            tree = ast.parse(formula_fix, mode="eval")
            _validate_formula_ast(tree)
            result_ttb_amount = eval(
                compile(tree, "<ex-ante-formula>", "eval"),
                {"__builtins__": {}},
                {
                    **ALLOWED_FORMULA_FUNCTIONS,
                    "e": math.e,
                    "pi": math.pi,
                },
            )
            return result_ttb_amount
        else:  # if we want to just check the text only (e.g comparing with treeo cloud ttbFormula!)
            return formula_fix


def _validate_formula_ast(tree):
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Mod,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported formula syntax: {node.__class__.__name__}")
        if isinstance(node, ast.Name) and node.id not in ALLOWED_FORMULA_NAMES:
            raise ValueError(f"Unsupported formula name: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Unsupported formula call.")
            if node.func.id not in ALLOWED_FORMULA_FUNCTIONS:
                raise ValueError(f"Unsupported formula function: {node.func.id}")
