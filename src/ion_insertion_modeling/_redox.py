"""Redox couples between two P_ox oxidation-state assignments.

P_ox guesses are per-element *average* oxidation states, so a partial change (Mn 4.0 -> 4.5) is an interval on
the oxidation-state axis. Each interval is split at integers and booked against the integer couples it spans.
Electrons are e⁻ per framework formula unit (the same basis as x), signed: > 0 when the element is oxidised on
charge, < 0 when it is reduced on charge ("counter redox"). Their sum equals z · delta_x when the guesses balance,
where z is the working ion's charge (1 for Li/Na, 2 for Mg).
"""
import math

from pymatgen.core import Composition

from ion_insertion_modeling import config

REDOX_COLUMNS = ["couples", "primary_couple", "n_couples", "electrons_total", "electron_residual",
                 "electron_balance_ok", "suspect_redox", "counter_redox", "redox_elements"]


# --- Couples ---
def integer_couples(ox_low: float, ox_high: float, eps: float = config.OXI_INTEGER_EPS) -> list[tuple[int, int, float]]:
    """Split the oxidation-state interval [ox_low, ox_high] at integers into (n, n+1, width) segments.
    4.0 -> 4.5 gives [(4, 5, 0.5)]; 3.5 -> 4.5 gives [(3, 4, 0.5), (4, 5, 0.5)]. Widths are e⁻ per atom."""
    segments = []
    n = math.floor(ox_low + eps)
    while n < ox_high - eps:
        width = min(ox_high, n + 1) - max(ox_low, n)
        if width > eps:
            segments.append((n, n + 1, width))
        n += 1
    return segments


def _format_state(n: int) -> str:
    return "0" if n == 0 else f"{abs(n)}{'+' if n > 0 else '-'}"


def couple_label(element: str, ox_low: int, ox_high: int) -> str:
    """E.g. "Mn3+/Mn4+", "O2-/O1-". Always written reduced/oxidised, whatever the direction of the step."""
    return f"{element}{_format_state(ox_low)}/{element}{_format_state(ox_high)}"


def element_amounts(framework_formula: str) -> dict[str, float]:
    """Atoms of each element per framework formula unit."""
    return {el.symbol: float(amt) for el, amt in Composition(framework_formula).items()}


def assign_couples(oxi_charge: dict, oxi_discharge: dict, framework_formula: str,
                   tol: float = config.OXI_CHANGE_TOL) -> list[dict]:
    """Couples active in one step. Each is {element, ox_from, ox_to, label, electrons, electrons_per_atom}, with
    ox_from < ox_to (integers), electrons in e⁻ per framework f.u. (negative if the element is reduced on charge)."""
    couples = []
    for el, amount in sorted(element_amounts(framework_formula).items()):
        if el not in oxi_charge or el not in oxi_discharge:
            raise ValueError(f"{framework_formula}: no oxidation state for {el} in one endpoint")
        ox_c, ox_d = float(oxi_charge[el]), float(oxi_discharge[el])
        if abs(ox_c - ox_d) <= tol:
            continue
        sign = 1.0 if ox_c > ox_d else -1.0
        for low, high, width in integer_couples(min(ox_c, ox_d), max(ox_c, ox_d)):
            couples.append({"element": el, "ox_from": low, "ox_to": high, "label": couple_label(el, low, high),
                            "electrons": sign * amount * width, "electrons_per_atom": sign * width})
    return couples


# --- Summaries ---
def check_balance(couples: list[dict], delta_x: float, tol: float = config.ELECTRON_BALANCE_TOL,
                  strict: bool = False, z: int = 1) -> bool:
    """True if the couples' net electrons equal z · delta_x (e⁻ per framework f.u.) within tol, where z is the
    working ion's charge. strict=True raises ValueError instead of returning False."""
    residual = sum(c["electrons"] for c in couples) - z * delta_x
    ok = abs(residual) <= tol
    if strict and not ok:
        raise ValueError(f"electron balance failed: residual {residual:+.4f} e⁻/f.u. (delta_x {delta_x}, z {z})")
    return ok


def primary_couple(couples: list[dict]) -> str | None:
    """Positive couple carrying the most electrons; ties go to the alphabetically first label."""
    positive = [c for c in couples if c["electrons"] > 0]
    if not positive:
        return None
    return min(positive, key=lambda c: (-round(c["electrons"], 9), c["label"]))["label"]


def summarise(couples: list[dict], delta_x: float,
              suspect_elements: tuple[str, ...] = config.SUSPECT_REDOX_ELEMENTS, z: int = 1) -> dict:
    """Per-pair redox columns (everything in REDOX_COLUMNS except `couples`). z = working ion charge."""
    total = sum(c["electrons"] for c in couples)
    elements = sorted({c["element"] for c in couples})
    return {
        "primary_couple": primary_couple(couples),
        "n_couples": len(couples),
        "electrons_total": total,
        "electron_residual": total - z * delta_x,
        "electron_balance_ok": bool(couples) and check_balance(couples, delta_x, z=z),
        "suspect_redox": any(el in suspect_elements for el in elements),
        "counter_redox": any(c["electrons"] < 0 for c in couples),
        "redox_elements": ",".join(elements),
    }
