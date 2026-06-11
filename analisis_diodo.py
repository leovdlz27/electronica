"""
Práctica 2 - Ingeniería Inversa de la No-Linealidad
Modelado de la curva I-V de un diodo real mediante la ecuación de Shockley.

Uso:
  1. Ingresa tus datos medidos en la sección DATA (voltaje y corriente).
  2. Ejecuta: python analisis_diodo.py
  3. Se generarán las gráficas y se estimarán los parámetros Is y n.

Repositorio: https://github.com/leovdlz27/electronica
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =============================================================================
# DATOS EXPERIMENTALES - Sustituir con tus mediciones reales
# Formato: voltaje en Voltios, corriente en Amperes
# Barrido de 0V a 1V en pasos de 0.05V
# =============================================================================
voltaje_medido = np.array([
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00
])

# Corriente medida con resistencia shunt (en mA convertido a A)
# REEMPLAZAR con tus valores reales del multímetro
corriente_medida = np.array([
    0.0000, 0.0000, 0.0000, 0.0001, 0.0002, 0.0005, 0.0012, 0.0025,
    0.0060, 0.0140, 0.0320, 0.0720, 0.1600, 0.3500, 0.7500, 1.5000,
    3.0000, 5.8000, 10.500, 18.000, 30.000
]) * 1e-3  # convertir mA a A si es necesario; ajustar según unidades reales


# =============================================================================
# MODELO DE SHOCKLEY
# I = Is * (exp(V / (n * Vt)) - 1)
# Vt = kT/q ≈ 0.02585 V a temperatura ambiente (25°C)
# =============================================================================
Vt = 0.02585  # Voltaje térmico a 25°C [V]

def shockley(V, Is, n):
    """Ecuación de Shockley para la curva I-V del diodo."""
    return Is * (np.exp(V / (n * Vt)) - 1)


# =============================================================================
# REGRESIÓN NO LINEAL - Ajuste de parámetros Is y n
# =============================================================================
# Filtrar solo puntos donde hay conducción apreciable (evitar log(0))
mascara = corriente_medida > 1e-9
V_fit = voltaje_medido[mascara]
I_fit = corriente_medida[mascara]

try:
    parametros, covarianza = curve_fit(
        shockley, V_fit, I_fit,
        p0=[1e-9, 1.5],           # estimados iniciales: Is~1nA, n~1.5
        bounds=([1e-15, 0.5], [1e-3, 2.5]),
        maxfev=10000
    )
    Is_estimado, n_estimado = parametros
    errores = np.sqrt(np.diag(covarianza))
    print("=" * 55)
    print("  RESULTADOS DE REGRESIÓN (Ecuación de Shockley)")
    print("=" * 55)
    print(f"  Corriente de saturación inversa  Is = {Is_estimado:.3e} A")
    print(f"  Factor de idealidad              n  = {n_estimado:.4f}")
    print(f"  Incertidumbre en Is: ±{errores[0]:.3e} A")
    print(f"  Incertidumbre en n:  ±{errores[1]:.4f}")
    print("=" * 55)
except RuntimeError as e:
    print(f"Error en el ajuste: {e}")
    Is_estimado, n_estimado = 1e-9, 1.5
    print("Usando valores iniciales por defecto.")


# =============================================================================
# GRÁFICA 1: Datos Experimentales vs Modelo de Shockley (escala lineal)
# =============================================================================
V_modelo = np.linspace(0, voltaje_medido.max(), 500)
I_modelo = shockley(V_modelo, Is_estimado, n_estimado)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Práctica 2 — Curva I-V del Diodo 1N4148", fontsize=14, fontweight='bold')

ax1 = axes[0]
ax1.plot(voltaje_medido * 1000, corriente_medida * 1000, 'o', color='steelblue',
         markersize=7, label='Datos experimentales', zorder=5)
ax1.plot(V_modelo * 1000, I_modelo * 1000, '-', color='tomato',
         linewidth=2, label=f'Modelo Shockley\nIs={Is_estimado:.2e} A, n={n_estimado:.3f}')
ax1.set_xlabel("Voltaje en el diodo $V_D$ [mV]", fontsize=11)
ax1.set_ylabel("Corriente $I_D$ [mA]", fontsize=11)
ax1.set_title("Escala Lineal", fontsize=12)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.35)
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)

# Pie de figura según formato requerido
ax1.text(0.01, -0.14,
         "Figura 1: Gráfica comparativa I-V del diodo 1N4148; "
         "los puntos representan mediciones experimentales y la línea el ajuste de Shockley.",
         transform=ax1.transAxes, fontsize=8, style='italic', wrap=True)

# =============================================================================
# GRÁFICA 2: Escala semilogarítmica para visualizar conducción exponencial
# =============================================================================
ax2 = axes[1]
mascara_pos = corriente_medida > 0
ax2.semilogy(voltaje_medido[mascara_pos] * 1000, corriente_medida[mascara_pos] * 1000,
             'o', color='steelblue', markersize=7, label='Datos experimentales', zorder=5)
ax2.semilogy(V_modelo * 1000, I_modelo * 1000, '-', color='tomato',
             linewidth=2, label='Modelo Shockley')
ax2.set_xlabel("Voltaje en el diodo $V_D$ [mV]", fontsize=11)
ax2.set_ylabel("Corriente $I_D$ [mA] (escala log)", fontsize=11)
ax2.set_title("Escala Semilogarítmica", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, which='both', alpha=0.35)

ax2.text(0.01, -0.14,
         "Figura 2: Representación semilogarítmica de la curva I-V; "
         "la linealidad confirma el comportamiento exponencial predicho por Shockley.",
         transform=ax2.transAxes, fontsize=8, style='italic', wrap=True)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("curva_iv_diodo.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n[OK] Gráfica guardada como 'curva_iv_diodo.png'")


# =============================================================================
# TABLA RESUMEN DE DATOS
# =============================================================================
print("\n  TABLA DE DATOS: Voltaje-Corriente")
print(f"  {'V_D [mV]':>10} | {'I_D [mA] (med)':>15} | {'I_D [mA] (modelo)':>18}")
print("  " + "-" * 48)
for v, i_med in zip(voltaje_medido, corriente_medida):
    i_mod = shockley(v, Is_estimado, n_estimado)
    print(f"  {v*1000:>10.1f} | {i_med*1000:>15.6f} | {i_mod*1000:>18.6f}")
