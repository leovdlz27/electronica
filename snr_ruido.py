"""
Práctica 5 - Caracterización de Ruido y SNR en Cadenas de Adquisición
Cálculo dinámico de SNR (dB) y ENOB bajo distintas condiciones de blindaje.

Modos de uso:
  Tiempo real:  python snr_ruido.py --puerto COM3
  CSV:          python snr_ruido.py --csv datos_ruido.csv
  Simulación:   python snr_ruido.py --simular

Formato CSV esperado: tiempo, señal_sin_blindaje, señal_con_blindaje
(los valores pueden ser en cuentas ADC o en voltios)

Repositorio: https://github.com/leovdlz27/electronica
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt
import argparse
import sys

# =============================================================================
# PARÁMETROS DEL SISTEMA
# =============================================================================
FS = 10_000      # Frecuencia de muestreo [Hz] del ADS1115 o ADC interno
BITS_ADC = 16    # Bits del ADS1115 (ajustar si usas ADC de 12 bits: 4095)
VREF = 3.3       # Voltaje de referencia del ADC [V]
LSB = VREF / (2**BITS_ADC)  # Paso de cuantización [V]


# =============================================================================
# CÁLCULOS DE SNR Y ENOB
# =============================================================================
def calcular_snr_db(señal, ruido):
    """SNR = 20·log10(RMS_señal / RMS_ruido)"""
    rms_señal = np.sqrt(np.mean(señal**2))
    rms_ruido = np.sqrt(np.mean(ruido**2))
    if rms_ruido < 1e-15:
        return float('inf')
    return 20 * np.log10(rms_señal / rms_ruido)


def calcular_enob(snr_db):
    """ENOB = (SNR_dB - 1.76) / 6.02"""
    return (snr_db - 1.76) / 6.02


def separar_señal_ruido(datos, fc=50, fs=FS):
    """
    Separa la señal útil (baja frecuencia) del ruido usando filtro paso-bajas.
    La señal filtrada es la 'señal limpia' y el residual es el 'ruido'.
    """
    nyq = fs / 2
    if fc >= nyq:
        fc = nyq * 0.9
    b, a = butter(4, fc / nyq, btype='low')
    señal_filtrada = filtfilt(b, a, datos)
    ruido_estimado = datos - señal_filtrada
    return señal_filtrada, ruido_estimado


def vpp(datos):
    """Voltaje pico a pico."""
    return np.max(datos) - np.min(datos)


# =============================================================================
# SIMULACIÓN DE ESCENARIOS
# =============================================================================
def simular_datos(n=5000, fs=FS):
    """
    Simula tres escenarios:
    1. Señal con batería (ruido bajo)
    2. Señal con fuente conmutada (ripple 60Hz + armónicos)
    3. Señal con blindaje capacitivo (ruido reducido)
    """
    t = np.arange(n) / fs
    señal_util = 1.5 + 0.5 * np.sin(2 * np.pi * 5 * t)  # señal a 5 Hz

    # Escenario 1: Batería (ruido gaussiano limpio)
    ruido_bateria = np.random.normal(0, 0.005, n)
    señal_bateria = señal_util + ruido_bateria

    # Escenario 2: Fuente conmutada (ripple 60Hz + intermodulación)
    ripple = (0.08 * np.sin(2 * np.pi * 60 * t) +
              0.04 * np.sin(2 * np.pi * 120 * t) +
              0.02 * np.sin(2 * np.pi * 180 * t))
    ruido_fuente = np.random.normal(0, 0.015, n)
    señal_fuente = señal_util + ripple + ruido_fuente

    # Escenario 3: Con blindaje (condensador de desacoplo, filtrado)
    b, a = butter(4, 200 / (fs/2), btype='low')
    señal_blindada = filtfilt(b, a, señal_fuente) + np.random.normal(0, 0.003, n)

    return t, señal_bateria, señal_fuente, señal_blindada


# =============================================================================
# GRAFICACIÓN Y REPORTE
# =============================================================================
def analizar_y_graficar(t, s_bateria, s_fuente, s_blindada, fs=FS):
    escenarios = {
        'Batería de Litio': s_bateria,
        'Fuente Conmutada (sin blindaje)': s_fuente,
        'Fuente Conmutada (con blindaje)': s_blindada,
    }
    colores = ['steelblue', 'tomato', 'forestgreen']

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Práctica 5 — Caracterización de Ruido, SNR y ENOB", fontsize=13, fontweight='bold')

    tabla_resultados = []

    for idx, (nombre, señal) in enumerate(escenarios.items()):
        señal_limpia, ruido_est = separar_señal_ruido(señal, fc=100, fs=fs)
        snr = calcular_snr_db(señal_limpia, ruido_est)
        enob = calcular_enob(snr)
        vpp_ruido = vpp(ruido_est)
        tabla_resultados.append((nombre, snr, enob, vpp_ruido * 1000))  # Vpp en mV

        # Panel izquierdo: señal en tiempo
        ax_t = axes[idx, 0]
        ax_t.plot(t[:2000], señal[:2000], color=colores[idx], linewidth=0.8, alpha=0.9)
        ax_t.set_xlabel("Tiempo [s]", fontsize=9)
        ax_t.set_ylabel("Voltaje [V]", fontsize=9)
        ax_t.set_title(f"{nombre}\nSNR={snr:.1f} dB | ENOB={enob:.2f} bits | Vpp_ruido={vpp_ruido*1000:.1f} mV",
                       fontsize=9)
        ax_t.grid(True, alpha=0.3)
        ax_t.text(0.01, -0.22,
            f"Figura {idx*2+1}: Señal capturada en escenario '{nombre}'; "
            f"la amplitud del ripple es visible en la envolvente de la forma de onda.",
            transform=ax_t.transAxes, fontsize=7, style='italic')

        # Panel derecho: PSD (espectro de potencia)
        ax_f = axes[idx, 1]
        freqs, psd = welch(señal, fs=fs, nperseg=min(1024, len(señal)//4))
        ax_f.semilogy(freqs, psd, color=colores[idx], linewidth=1.0)
        ax_f.set_xlabel("Frecuencia [Hz]", fontsize=9)
        ax_f.set_ylabel("PSD [V²/Hz]", fontsize=9)
        ax_f.set_title(f"Densidad Espectral de Potencia — {nombre}", fontsize=9)
        ax_f.grid(True, which='both', alpha=0.3)
        # Marcar pico de ripple en 60Hz si aplica
        if 'Conmutada' in nombre:
            ax_f.axvline(60, color='red', linestyle=':', alpha=0.7, linewidth=1.2, label='60 Hz')
            ax_f.axvline(120, color='orange', linestyle=':', alpha=0.7, linewidth=1.2, label='120 Hz')
            ax_f.legend(fontsize=7)
        ax_f.text(0.01, -0.22,
            f"Figura {idx*2+2}: PSD del escenario '{nombre}'; "
            "los picos espectrales identifican las fuentes de interferencia electromagnética.",
            transform=ax_f.transAxes, fontsize=7, style='italic')

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig("analisis_ruido.png", dpi=150, bbox_inches='tight')
    plt.show()

    # ==========================================================================
    # TABLA RESUMEN
    # ==========================================================================
    print("\n" + "=" * 72)
    print("  TABLA COMPARATIVA — SNR, ENOB y Vpp de Ruido por Escenario")
    print("=" * 72)
    print(f"  {'Escenario':<38} | {'SNR [dB]':>9} | {'ENOB [bits]':>11} | {'Vpp Ruido [mV]':>14}")
    print("  " + "-" * 72)
    for nombre, snr, enob, vpp_mv in tabla_resultados:
        print(f"  {nombre:<38} | {snr:>9.2f} | {enob:>11.2f} | {vpp_mv:>14.3f}")
    print("=" * 72)
    print(f"\n  ADC teórico: {BITS_ADC} bits | Vref: {VREF}V | LSB: {LSB*1000:.4f} mV")
    print(f"  Frecuencia de muestreo: {FS} Hz")
    print("\n[OK] Gráfica guardada como 'analisis_ruido.png'")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Práctica 5 — SNR y ENOB")
    parser.add_argument('--puerto',   type=str, help='Puerto serial')
    parser.add_argument('--baudrate', type=int, default=115200)
    parser.add_argument('--csv',      type=str, help='CSV con columnas: tiempo,s_bateria,s_fuente,s_blindada')
    parser.add_argument('--simular',  action='store_true')
    args = parser.parse_args()

    if args.csv:
        datos = np.genfromtxt(args.csv, delimiter=',', skip_header=1)
        t, s_bat, s_fue, s_bli = datos[:, 0], datos[:, 1], datos[:, 2], datos[:, 3]
        analizar_y_graficar(t, s_bat, s_fue, s_bli)
    else:
        if not args.simular:
            print("[INFO] No se especificó puerto ni CSV. Modo simulación activado.")
        t, s_bat, s_fue, s_bli = simular_datos()
        analizar_y_graficar(t, s_bat, s_fue, s_bli)
