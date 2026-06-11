"""
Práctica 3 - El Perceptrón Analógico (Hardware)
Simulación y validación del sumador ponderado: y = sum(wi * xi) + b

Uso:
  1. Modifica las listas de entradas y pesos según tu diseño de resistencias.
  2. Ejecuta: python simulacion_perceptron.py
  3. Se generan la tabla comparativa y las gráficas de validación.

Repositorio: https://github.com/leovdlz27/electronica
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

# =============================================================================
# PARÁMETROS DEL CIRCUITO - Ajustar según tu diseño
# Sumador inversor con LM358
# Ganancia de cada entrada: Ai = -Rf / Ri
# =============================================================================
Rf = 10_000   # Resistencia de retroalimentación [Ω]  — ajustar a tu valor
R1 = 10_000   # Resistencia entrada x1 → peso w1 = Rf/R1
R2 = 20_000   # Resistencia entrada x2 → peso w2 = Rf/R2
R_bias = 10_000  # Resistencia del potenciómetro de bias

# Pesos calculados (ganancia inversora, signo negativo incluido luego)
w1 = Rf / R1    # = 1.0
w2 = Rf / R2    # = 0.5

# Voltaje de alimentación del LM358 (límites de saturación)
VCC_pos = 5.0   # [V]
VCC_neg = 0.0   # [V] (alimentación single-supply)

print("=" * 55)
print("  CONFIGURACIÓN DEL SUMADOR PONDERADO (LM358)")
print("=" * 55)
print(f"  Rf = {Rf/1000:.1f} kΩ")
print(f"  R1 = {R1/1000:.1f} kΩ  →  w1 = {w1:.3f}")
print(f"  R2 = {R2/1000:.1f} kΩ  →  w2 = {w2:.3f}")
print(f"  Alimentación: {VCC_neg}V a {VCC_pos}V")
print("=" * 55)


# =============================================================================
# FUNCIÓN DEL SUMADOR INVERSOR IDEAL
# y_ideal = -(w1*x1 + w2*x2 + w_b*b)
# El LM358 satura entre VCC_neg y VCC_pos (aprox ±1.5V del riel)
# =============================================================================
def sumador_ideal(x1, x2, b, w1, w2):
    """Salida teórica del sumador inversor (sin saturación)."""
    return -(w1 * x1 + w2 * x2 + b)

def sumador_saturado(x1, x2, b, w1, w2, vcc_min=VCC_neg + 0.1, vcc_max=VCC_pos - 0.1):
    """Salida teórica con clipping por límites del amplificador."""
    y = sumador_ideal(x1, x2, b, w1, w2)
    return np.clip(y, vcc_min, vcc_max)


# =============================================================================
# COMBINACIONES DE ENTRADA PARA TABLA DE VALIDACIÓN
# xi ∈ {0V, 2.5V, 5V} — voltajes representativos de lógica digital
# REEMPLAZAR salida_medida con tus lecturas reales del osciloscopio
# =============================================================================
entradas_x1 = [0.0, 0.0, 2.5, 2.5, 5.0, 5.0, 0.0, 5.0]
entradas_x2 = [0.0, 2.5, 0.0, 2.5, 0.0, 2.5, 5.0, 5.0]
bias_b      = [0.5] * 8   # Potenciómetro fijo en 0.5V

# REEMPLAZAR con mediciones reales del multímetro/osciloscopio
salida_medida = [None] * 8  # ← pon aquí tus valores: ej. [-0.50, -1.72, ...]

# Calcular salida teórica para cada combinación
salida_teorica = [
    sumador_saturado(x1, x2, b, w1, w2)
    for x1, x2, b in zip(entradas_x1, entradas_x2, bias_b)
]

print("\n  TABLA COMPARATIVA — Entradas vs Salida Teórica vs Medida")
print(f"  {'x1 [V]':>8} | {'x2 [V]':>8} | {'b [V]':>7} | "
      f"{'y_teorica [V]':>14} | {'y_medida [V]':>13} | {'Error %':>8}")
print("  " + "-" * 70)
for x1, x2, b, y_t, y_m in zip(entradas_x1, entradas_x2, bias_b, salida_teorica, salida_medida):
    if y_m is not None:
        err = abs((y_m - y_t) / y_t * 100) if y_t != 0 else float('nan')
        print(f"  {x1:>8.2f} | {x2:>8.2f} | {b:>7.2f} | {y_t:>14.4f} | {y_m:>13.4f} | {err:>7.2f}%")
    else:
        print(f"  {x1:>8.2f} | {x2:>8.2f} | {b:>7.2f} | {y_t:>14.4f} | {'(pendiente)':>13} | {'---':>8}")


# =============================================================================
# GRÁFICA 1: Salida teórica vs entradas (mapa de calor del espacio de entrada)
# =============================================================================
x1_range = np.linspace(0, 5, 200)
x2_range = np.linspace(0, 5, 200)
X1, X2 = np.meshgrid(x1_range, x2_range)
B_fixed = 0.5  # bias fijo para la visualización

Y_ideal = sumador_ideal(X1, X2, B_fixed, w1, w2)
Y_sat   = sumador_saturado(X1, X2, B_fixed, w1, w2)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Práctica 3 — Perceptrón Analógico: Sumador Ponderado LM358",
             fontsize=13, fontweight='bold')

im1 = axes[0].contourf(X1, X2, Y_ideal, levels=30, cmap='RdBu_r')
plt.colorbar(im1, ax=axes[0], label='Salida y [V]')
axes[0].set_xlabel("Entrada $x_1$ [V]", fontsize=11)
axes[0].set_ylabel("Entrada $x_2$ [V]", fontsize=11)
axes[0].set_title("Salida Teórica (sin saturación)", fontsize=11)
axes[0].text(0.01, -0.16,
    "Figura 1: Mapa de la función y = -(w₁·x₁ + w₂·x₂ + b) para b=0.5V; "
    "el gradiente representa el espacio de decisión lineal del perceptrón.",
    transform=axes[0].transAxes, fontsize=8, style='italic')

im2 = axes[1].contourf(X1, X2, Y_sat, levels=30, cmap='RdBu_r')
plt.colorbar(im2, ax=axes[1], label='Salida y [V]')
axes[1].set_xlabel("Entrada $x_1$ [V]", fontsize=11)
axes[1].set_ylabel("Entrada $x_2$ [V]", fontsize=11)
axes[1].set_title(f"Salida con Saturación (rails: {VCC_neg}V–{VCC_pos}V)", fontsize=11)
axes[1].text(0.01, -0.16,
    "Figura 2: Efecto del clipping por límites del LM358; "
    "las zonas saturadas evidencian la restricción física del rango dinámico del amplificador.",
    transform=axes[1].transAxes, fontsize=8, style='italic')

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("mapa_perceptron.png", dpi=150, bbox_inches='tight')
plt.show()


# =============================================================================
# GRÁFICA 2: Barrido de bias — efecto del potenciómetro sobre la salida
# =============================================================================
bias_range = np.linspace(-2.5, 2.5, 300)
x1_fijo, x2_fijo = 2.5, 2.5  # entradas fijas para demostración

y_bias = [sumador_saturado(x1_fijo, x2_fijo, b, w1, w2) for b in bias_range]

fig2, ax = plt.subplots(figsize=(9, 5))
ax.plot(bias_range, y_bias, color='steelblue', linewidth=2.5, label='Salida y(b)')
ax.axhline(y=VCC_pos - 0.1, color='tomato', linestyle='--', alpha=0.7, label='Límite superior LM358')
ax.axhline(y=VCC_neg + 0.1, color='tomato', linestyle='--', alpha=0.7, label='Límite inferior LM358')
ax.set_xlabel("Voltaje de Bias $b$ [V]", fontsize=11)
ax.set_ylabel("Salida $y$ [V]", fontsize=11)
ax.set_title(f"Efecto del Bias en la Salida ($x_1$={x1_fijo}V, $x_2$={x2_fijo}V)", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.35)
ax.text(0.01, -0.14,
    "Figura 3: Desplazamiento de offset en función del potenciómetro de bias; "
    "la región plana indica saturación del amplificador operacional.",
    transform=ax.transAxes, fontsize=8, style='italic')

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("efecto_bias.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n[OK] Gráficas guardadas: 'mapa_perceptron.png', 'efecto_bias.png'")
