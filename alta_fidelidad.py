"""
Práctica 6 - Digitalización de Alta Fidelidad
Verificación de Nyquist, análisis espectral y preprocesamiento para Edge Computing.

Funciones:
  1. Diseño matemático del filtro Sallen-Key (calcula R y C)
  2. Verificación espectral de aliasing antes/después del filtro
  3. Preprocesamiento Min-Max Scaling para TinyML
  4. Análisis de latencia y métricas del pipeline

Modos:
  python alta_fidelidad.py --simular
  python alta_fidelidad.py --csv datos_audio.csv --fs 10000

Repositorio: https://github.com/leovdlz27/electronica
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz, welch
from scipy.fft import fft, fftfreq
import argparse
import time

# =============================================================================
# PARÁMETROS DEL SISTEMA - Ajustar según tu diseño
# =============================================================================
FS_MUESTREO = 10_000   # Frecuencia de muestreo del ADC [Hz]
FC_CORTE    = 4_000    # Frecuencia de corte del filtro anti-aliasing [Hz]
                       # Regla de Nyquist: fc < fs/2 = 5000 Hz
ORDEN_FILTRO = 2       # Sallen-Key de 2do orden (Butterworth)
VREF = 3.3             # Voltaje de referencia [V]
BITS = 12              # Bits del ADC (12 para ESP32, 16 para ADS1115)


# =============================================================================
# DISEÑO DEL FILTRO SALLEN-KEY
# Para Butterworth de 2do orden con ganancia unitaria:
#   fc = 1 / (2π · RC)
#   R1 = R2 = R;  C1 = C2 = C  (configuración simplificada)
# =============================================================================
def diseñar_sallen_key(fc, tipo='Butterworth'):
    """
    Calcula los componentes R y C para un filtro Sallen-Key de 2do orden.
    Devuelve los valores estándar más cercanos.
    """
    # Series estándar E24 de resistencias y capacitores
    E24_R = np.array([1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4,
                      2.7, 3.0, 3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2,
                      6.8, 7.5, 8.2, 9.1]) * 1000  # kΩ × multiplicadores

    C_objetivo = 10e-9  # 10 nF (valor de partida común)
    R_calculada = 1 / (2 * np.pi * fc * C_objetivo)

    # Valor más cercano en E24 × 1kΩ, 10kΩ, 100kΩ
    for mult in [1, 10, 100]:
        candidatos = E24_R * mult
        idx = np.argmin(np.abs(candidatos - R_calculada))
        R_std = candidatos[idx]
        if 1e3 <= R_std <= 1e6:  # entre 1kΩ y 1MΩ
            break

    fc_real = 1 / (2 * np.pi * R_std * C_objetivo)

    print("=" * 60)
    print("  DISEÑO DEL FILTRO SALLEN-KEY (2° Orden Butterworth)")
    print("=" * 60)
    print(f"  Frecuencia de corte deseada:  fc = {fc/1000:.2f} kHz")
    print(f"  Frecuencia de muestreo:       fs = {FS_MUESTREO/1000:.1f} kHz")
    print(f"  Margen Nyquist (fs/2):             {FS_MUESTREO/2000:.1f} kHz ✓")
    print(f"  ─────────────────────────────────────────")
    print(f"  Capacitor elegido:    C = {C_objetivo*1e9:.0f} nF (estándar)")
    print(f"  Resistencia calculada: R = {R_calculada/1000:.2f} kΩ")
    print(f"  Resistencia estándar:  R = {R_std/1000:.2f} kΩ (E24)")
    print(f"  Frecuencia real:      fc_real = {fc_real/1000:.3f} kHz")
    print(f"  Error:                {abs(fc_real-fc)/fc*100:.2f}%")
    print("=" * 60)

    return R_std, C_objetivo, fc_real


# =============================================================================
# PREPROCESAMIENTO MIN-MAX SCALING (para TinyML / Edge Computing)
# =============================================================================
def min_max_scaling(datos, rango_salida=(0.0, 1.0)):
    """
    Normalización Min-Max: x_norm = (x - x_min) / (x_max - x_min)
    Escala a [rango_salida[0], rango_salida[1]].
    Libera carga computacional en la nube al enviar datos ya normalizados.
    """
    d_min, d_max = np.min(datos), np.max(datos)
    if d_max == d_min:
        return np.zeros_like(datos), d_min, d_max
    norm = (datos - d_min) / (d_max - d_min)
    a, b = rango_salida
    return a + norm * (b - a), d_min, d_max


# =============================================================================
# SIMULACIÓN DE SEÑAL CON Y SIN ALIASING
# =============================================================================
def simular_señal(fs=FS_MUESTREO, fc_filtro=FC_CORTE, duracion=0.5):
    """
    Genera:
    - Señal 'sucia': señal útil + componente de alta frecuencia (alias)
    - Señal filtrada: después del filtro anti-aliasing Sallen-Key
    """
    n = int(fs * duracion)
    t = np.arange(n) / fs

    # Señal útil (dentro de banda)
    f_util = 1000   # 1 kHz — señal de audio/sensor
    # Componente que causaría aliasing si no se filtra
    f_alias = fs * 0.7  # 7000 Hz > Nyquist de 5000 Hz

    señal_util = 0.8 * np.sin(2 * np.pi * f_util * t)
    señal_alias = 0.3 * np.sin(2 * np.pi * f_alias * t)
    ruido = np.random.normal(0, 0.02, n)

    señal_cruda = señal_util + señal_alias + ruido

    # Aplicar filtro Sallen-Key (Butterworth 2do orden)
    nyq = fs / 2
    b, a = butter(ORDEN_FILTRO, fc_filtro / nyq, btype='low')
    señal_filtrada = filtfilt(b, a, señal_cruda)

    return t, señal_cruda, señal_filtrada, f_util, f_alias


# =============================================================================
# ANÁLISIS ESPECTRAL
# =============================================================================
def espectro(señal, fs):
    n = len(señal)
    Y = np.abs(fft(señal))[:n//2] * 2/n
    freqs = fftfreq(n, 1/fs)[:n//2]
    return freqs, Y


# =============================================================================
# GRAFICACIÓN COMPLETA
# =============================================================================
def graficar(t, señal_cruda, señal_filtrada, f_util, f_alias, R, C, fc_real):
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Práctica 6 — Digitalización de Alta Fidelidad (Anti-Aliasing + Nyquist)",
                 fontsize=13, fontweight='bold')

    # --- 1. Señal en tiempo: Cruda vs Filtrada ---
    ax1 = fig.add_subplot(3, 2, (1, 2))
    n_show = min(1000, len(t))
    ax1.plot(t[:n_show]*1000, señal_cruda[:n_show], color='tomato', linewidth=0.9,
             alpha=0.8, label='Señal sin filtrar (con componente de alias)')
    ax1.plot(t[:n_show]*1000, señal_filtrada[:n_show], color='steelblue', linewidth=1.5,
             label=f'Señal filtrada (Sallen-Key fc={fc_real/1000:.2f} kHz)')
    ax1.set_xlabel("Tiempo [ms]", fontsize=10)
    ax1.set_ylabel("Amplitud [V]", fontsize=10)
    ax1.set_title("Señal Antes y Después del Filtro Anti-Aliasing", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.01, -0.12,
        "Figura 1: Comparativa temporal señal cruda vs filtrada; "
        "el filtro Sallen-Key elimina las componentes de alta frecuencia que causarían aliasing espectral.",
        transform=ax1.transAxes, fontsize=8, style='italic')

    # --- 2. Espectro: Verificación de aliasing ---
    ax2 = fig.add_subplot(3, 2, 3)
    f_cruda, Y_cruda = espectro(señal_cruda, FS_MUESTREO)
    ax2.semilogy(f_cruda/1000, Y_cruda + 1e-6, color='tomato', linewidth=1.0, alpha=0.85)
    ax2.axvline(f_util/1000, color='green', linestyle='--', linewidth=1.3,
                label=f'Señal útil ({f_util/1000:.1f} kHz)')
    ax2.axvline(f_alias/1000, color='red', linestyle=':', linewidth=1.3,
                label=f'Alias ({f_alias/1000:.1f} kHz)')
    ax2.axvline(FS_MUESTREO/2/1000, color='black', linestyle='-', linewidth=1.2, alpha=0.5,
                label=f'Nyquist ({FS_MUESTREO/2/1000:.1f} kHz)')
    ax2.set_xlabel("Frecuencia [kHz]", fontsize=10)
    ax2.set_ylabel("|FFT| (log)", fontsize=10)
    ax2.set_title("Espectro Señal SIN Filtrar", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, which='both', alpha=0.3)
    ax2.text(0.01, -0.18,
        "Figura 2: Espectro de la señal cruda; la componente de alias supera el límite de Nyquist "
        "y se plegará sobre la banda útil si no se filtra antes de digitalizar.",
        transform=ax2.transAxes, fontsize=8, style='italic')

    # --- 3. Espectro filtrado: Sin aliasing ---
    ax3 = fig.add_subplot(3, 2, 4)
    f_filt, Y_filt = espectro(señal_filtrada, FS_MUESTREO)
    ax3.semilogy(f_filt/1000, Y_filt + 1e-6, color='steelblue', linewidth=1.0)
    ax3.axvline(f_util/1000, color='green', linestyle='--', linewidth=1.3,
                label=f'Señal útil ({f_util/1000:.1f} kHz)')
    ax3.axvline(fc_real/1000, color='purple', linestyle='-.', linewidth=1.3,
                label=f'fc filtro ({fc_real/1000:.2f} kHz)')
    ax3.axvline(FS_MUESTREO/2/1000, color='black', linestyle='-', linewidth=1.2, alpha=0.5,
                label=f'Nyquist ({FS_MUESTREO/2/1000:.1f} kHz)')
    ax3.set_xlabel("Frecuencia [kHz]", fontsize=10)
    ax3.set_ylabel("|FFT| (log)", fontsize=10)
    ax3.set_title("Espectro Señal CON Filtro (Anti-Aliasing)", fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(True, which='both', alpha=0.3)
    ax3.text(0.01, -0.18,
        "Figura 3: Espectro post-filtrado; la ausencia de componentes sobre fc confirma "
        "el cumplimiento del teorema de Nyquist-Shannon en la cadena de adquisición.",
        transform=ax3.transAxes, fontsize=8, style='italic')

    # --- 4. Respuesta en frecuencia del filtro diseñado ---
    ax4 = fig.add_subplot(3, 2, 5)
    nyq = FS_MUESTREO / 2
    b, a = butter(ORDEN_FILTRO, FC_CORTE / nyq, btype='low')
    w, h = freqz(b, a, worN=2000)
    freqs_hz = w / np.pi * nyq
    ax4.semilogx(freqs_hz/1000, 20*np.log10(np.abs(h) + 1e-10), color='purple', linewidth=2)
    ax4.axvline(FC_CORTE/1000, color='red', linestyle='--', linewidth=1.3,
                label=f'fc = {FC_CORTE/1000:.1f} kHz')
    ax4.axhline(-3, color='gray', linestyle=':', linewidth=1, label='-3 dB')
    ax4.set_xlabel("Frecuencia [kHz]", fontsize=10)
    ax4.set_ylabel("Atenuación [dB]", fontsize=10)
    ax4.set_title("Respuesta en Frecuencia del Filtro Sallen-Key", fontsize=11)
    ax4.legend(fontsize=8)
    ax4.grid(True, which='both', alpha=0.3)
    ax4.set_ylim(-80, 5)
    ax4.text(0.01, -0.18,
        f"Figura 4: Diagrama de Bode del filtro Sallen-Key diseñado (R={R/1000:.1f}kΩ, C={C*1e9:.0f}nF); "
        "la pendiente de -40 dB/dec atenúa efectivamente las componentes de alias.",
        transform=ax4.transAxes, fontsize=8, style='italic')

    # --- 5. Normalización Min-Max para Edge Computing ---
    ax5 = fig.add_subplot(3, 2, 6)
    t_show = t[:500]
    señal_norm, v_min, v_max = min_max_scaling(señal_filtrada[:500])
    ax5.plot(t_show*1000, señal_filtrada[:500], color='steelblue', linewidth=1.2,
             label='Señal filtrada [V]', alpha=0.8)
    ax5_b = ax5.twinx()
    ax5_b.plot(t_show*1000, señal_norm, color='darkorange', linewidth=1.2,
               label='Normalizada [0,1]', alpha=0.8)
    ax5.set_xlabel("Tiempo [ms]", fontsize=10)
    ax5.set_ylabel("Voltaje [V]", color='steelblue', fontsize=10)
    ax5_b.set_ylabel("Valor normalizado", color='darkorange', fontsize=10)
    ax5.set_title("Preprocesamiento Edge: Min-Max Scaling", fontsize=11)
    # Leyendas combinadas
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_b.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.text(0.01, -0.18,
        f"Figura 5: Normalización Min-Max en el microcontrolador (x_min={v_min:.3f}V, x_max={v_max:.3f}V); "
        "reducir a [0,1] antes de transmitir disminuye la carga computacional en la nube/PC.",
        transform=ax5.transAxes, fontsize=8, style='italic')

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig("alta_fidelidad.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[OK] Gráfica guardada como 'alta_fidelidad.png'")

    # --- Métricas finales ---
    señal_n, _, _ = min_max_scaling(señal_filtrada)
    print("\n" + "=" * 60)
    print("  MÉTRICAS DEL PIPELINE DE ADQUISICIÓN")
    print("=" * 60)
    print(f"  Frecuencia de muestreo:      {FS_MUESTREO/1000:.1f} kHz")
    print(f"  Frecuencia de corte (fc):    {fc_real/1000:.3f} kHz")
    print(f"  Límite de Nyquist (fs/2):    {FS_MUESTREO/2/1000:.1f} kHz")
    print(f"  Margen anti-aliasing:        {(FS_MUESTREO/2 - fc_real)/1000:.3f} kHz")
    print(f"  Resolución ADC:              {BITS} bits ({2**BITS} niveles)")
    lsb_mv = VREF / (2**BITS) * 1000
    print(f"  LSB equivalente:             {lsb_mv:.4f} mV")
    print(f"  Rango normalizado:           [{señal_n.min():.4f}, {señal_n.max():.4f}]")
    print("=" * 60)


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Práctica 6 — Alta Fidelidad")
    parser.add_argument('--csv',     type=str, help='CSV con columna de señal')
    parser.add_argument('--fs',      type=int, default=FS_MUESTREO, help='Frecuencia de muestreo [Hz]')
    parser.add_argument('--simular', action='store_true')
    args = parser.parse_args()

    # Diseñar filtro
    R, C, fc_real = diseñar_sallen_key(FC_CORTE)

    if args.csv:
        datos = np.genfromtxt(args.csv, delimiter=',', skip_header=1)
        señal_cruda = datos[:, 1]
        t = datos[:, 0]
        nyq = args.fs / 2
        b, a = butter(ORDEN_FILTRO, FC_CORTE / nyq, btype='low')
        señal_filtrada = filtfilt(b, a, señal_cruda)
        graficar(t, señal_cruda, señal_filtrada, 1000, args.fs*0.7, R, C, fc_real)
    else:
        print("[INFO] Modo simulación.")
        t, s_cruda, s_filt, f_util, f_alias = simular_señal()
        graficar(t, s_cruda, s_filt, f_util, f_alias, R, C, fc_real)
