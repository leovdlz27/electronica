"""
Práctica 4 - Fusión de Sensores y Calibración
Adquisición serial desde ESP32/Arduino, gráficas temporales y análisis de correlación.

Modos de uso:
  A) Datos en tiempo real (requiere hardware conectado):
       python fusion_sensores.py --puerto COM3 --baudrate 115200

  B) Datos de archivo CSV (modo offline / simulación):
       python fusion_sensores.py --csv datos_sensores.csv

  C) Simulación interna (sin hardware ni archivo):
       python fusion_sensores.py --simular

Repositorio: https://github.com/leovdlz27/electronica
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import time

# =============================================================================
# PARÁMETROS DE CALIBRACIÓN - Ajustar con tus mediciones reales
# Función de transferencia: Lux = m * ADC + c  (ajuste lineal/log)
# =============================================================================
# Puntos de calibración medidos (ADC crudo vs. luxómetro de referencia)
CALIBRACION_ADC  = [0,    868,  1089, 1443]   # valores ADC del ADS1115
CALIBRACION_LUX  = [0,    50,   200,  1000]   # luxes del instrumento de referencia

# Ajuste de calibración (se calcula automáticamente)
coef_cal = np.polyfit(CALIBRACION_ADC, CALIBRACION_LUX, deg=1)  # lineal


def adc_a_lux(adc_val):
    """Convierte lectura ADC cruda a Luxes usando la curva de calibración."""
    return np.polyval(coef_cal, adc_val)


def adc_a_celsius(adc_val, vcc=3.3, bits=16):
    """
    Conversión simplificada para termistor NTC o LM35.
    LM35: Vout = 10mV/°C → T = (ADC/max_ADC * VCC) / 0.01
    Ajustar según tu sensor real.
    """
    voltaje = (adc_val / (2**bits - 1)) * vcc
    temperatura = voltaje / 0.01  # para LM35
    return temperatura


# =============================================================================
# CÁLCULO DE ENOB (Número Efectivo de Bits)
# ENOB = (SNR_dB - 1.76) / 6.02
# =============================================================================
def calcular_enob(señal, ruido_rms=None):
    """Estima ENOB a partir de la señal capturada."""
    señal = np.array(señal)
    if ruido_rms is None:
        # Estimar ruido como desviación estándar del residual de una media móvil
        ventana = min(10, len(señal) // 5)
        if ventana < 2:
            return None
        media_movil = np.convolve(señal, np.ones(ventana)/ventana, mode='valid')
        residual = señal[:len(media_movil)] - media_movil
        ruido_rms = np.std(residual)
    señal_rms = np.std(señal)
    if ruido_rms < 1e-12:
        return float('inf')
    snr_lineal = señal_rms / ruido_rms
    snr_db = 20 * np.log10(snr_lineal) if snr_lineal > 0 else 0
    enob = (snr_db - 1.76) / 6.02
    return max(enob, 0), snr_db


# =============================================================================
# MODO A: ADQUISICIÓN SERIAL EN TIEMPO REAL
# =============================================================================
def adquirir_serial(puerto, baudrate=115200, n_muestras=500):
    """Lee datos del formato 'LUZ:valor,TEMP:valor\\n' desde UART."""
    try:
        import serial
    except ImportError:
        print("[ERROR] Instala pyserial: pip install pyserial")
        sys.exit(1)

    luz_raw, temp_raw, timestamps = [], [], []
    print(f"[INFO] Conectando a {puerto} @ {baudrate} baud...")
    try:
        ser = serial.Serial(puerto, baudrate, timeout=2)
        time.sleep(2)
        print(f"[INFO] Leyendo {n_muestras} muestras...")
        t0 = time.time()
        while len(luz_raw) < n_muestras:
            linea = ser.readline().decode('utf-8', errors='ignore').strip()
            if 'LUZ:' in linea and 'TEMP:' in linea:
                try:
                    partes = linea.split(',')
                    lux_val = float(partes[0].split(':')[1])
                    tmp_val = float(partes[1].split(':')[1])
                    luz_raw.append(lux_val)
                    temp_raw.append(tmp_val)
                    timestamps.append(time.time() - t0)
                except (ValueError, IndexError):
                    continue
        ser.close()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    return np.array(timestamps), np.array(luz_raw), np.array(temp_raw)


# =============================================================================
# MODO B: CARGA DESDE CSV
# =============================================================================
def cargar_csv(ruta_csv):
    """Carga datos desde archivo CSV con columnas: tiempo, luz_adc, temp_adc"""
    datos = np.genfromtxt(ruta_csv, delimiter=',', skip_header=1)
    return datos[:, 0], datos[:, 1], datos[:, 2]


# =============================================================================
# MODO C: SIMULACIÓN INTERNA
# =============================================================================
def simular_datos(n=500, fs=10):
    """Genera datos sintéticos que emulan el comportamiento real de los sensores."""
    t = np.arange(n) / fs
    # Simular variación ambiental + ruido del ADC
    lux_base = 800 + 200 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 15, n)
    temp_base = 25 + 5 * np.sin(2 * np.pi * t / 60 + 0.5) + np.random.normal(0, 0.8, n)
    # Simular 3 condiciones (oscuro, normal, brillante)
    lux_base[:n//3] *= 0.1
    lux_base[2*n//3:] *= 1.8
    lux_base = np.clip(lux_base, 0, 1600)
    return t, lux_base, temp_base


# =============================================================================
# GRAFICACIÓN Y ANÁLISIS
# =============================================================================
def graficar_y_analizar(t, luz_adc, temp_raw):
    lux = adc_a_lux(luz_adc)
    temp = adc_a_celsius(temp_raw) if temp_raw.max() > 100 else temp_raw  # si ya está en °C

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Práctica 4 — Fusión de Sensores: LDR y Temperatura", fontsize=14, fontweight='bold')

    # --- Gráfica 1: Series temporales superpuestas ---
    ax1 = fig.add_subplot(2, 2, 1)
    color_lux = 'darkorange'
    color_tmp = 'steelblue'
    ax1.set_xlabel("Tiempo [s]", fontsize=10)
    ax1.set_ylabel("Iluminancia [Lux]", color=color_lux, fontsize=10)
    l1, = ax1.plot(t, lux, color=color_lux, linewidth=1.2, alpha=0.85, label='Luz (Lux)')
    ax1.tick_params(axis='y', labelcolor=color_lux)

    ax1b = ax1.twinx()
    ax1b.set_ylabel("Temperatura [°C]", color=color_tmp, fontsize=10)
    l2, = ax1b.plot(t, temp, color=color_tmp, linewidth=1.2, alpha=0.85, label='Temperatura (°C)')
    ax1b.tick_params(axis='y', labelcolor=color_tmp)

    ax1.set_title("Series Temporales Superpuestas", fontsize=11)
    ax1.legend(handles=[l1, l2], loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.01, -0.18,
        "Figura 1: Evolución temporal de ambos sensores; la correlación temporal es visible "
        "en la respuesta común ante cambios del entorno.",
        transform=ax1.transAxes, fontsize=7, style='italic')

    # --- Gráfica 2: Scatter Plot Luz vs Temperatura ---
    ax2 = fig.add_subplot(2, 2, 2)
    sc = ax2.scatter(lux, temp, c=t, cmap='plasma', s=12, alpha=0.6)
    plt.colorbar(sc, ax=ax2, label='Tiempo [s]')
    # Línea de tendencia
    z = np.polyfit(lux, temp, 1)
    p = np.poly1d(z)
    lux_sorted = np.sort(lux)
    ax2.plot(lux_sorted, p(lux_sorted), 'r--', linewidth=1.5,
             label=f'Tendencia: y={z[0]:.4f}x+{z[1]:.2f}')
    # Correlación de Pearson
    r = np.corrcoef(lux, temp)[0, 1]
    ax2.set_xlabel("Iluminancia [Lux]", fontsize=10)
    ax2.set_ylabel("Temperatura [°C]", fontsize=10)
    ax2.set_title(f"Scatter Plot: Luz vs Temperatura (r={r:.3f})", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.text(0.01, -0.18,
        f"Figura 2: Diagrama de dispersión con r={r:.3f}; el coeficiente de Pearson cuantifica "
        "la correlación ambiental entre irradiancia lumínica y temperatura.",
        transform=ax2.transAxes, fontsize=7, style='italic')

    # --- Gráfica 3: Curva de calibración LDR ---
    ax3 = fig.add_subplot(2, 2, 3)
    adc_cal = np.array(CALIBRACION_ADC)
    lux_cal = np.array(CALIBRACION_LUX)
    adc_fine = np.linspace(0, adc_cal.max() * 1.1, 300)
    ax3.scatter(adc_cal, lux_cal, color='forestgreen', s=80, zorder=5, label='Puntos de calibración')
    ax3.plot(adc_fine, np.polyval(coef_cal, adc_fine), 'g--', linewidth=1.8,
             label=f'Ajuste lineal: y={coef_cal[0]:.4f}·ADC+{coef_cal[1]:.2f}')
    ax3.set_xlabel("Valor ADC crudo", fontsize=10)
    ax3.set_ylabel("Iluminancia Real [Lux]", fontsize=10)
    ax3.set_title("Curva de Calibración LDR", fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.text(0.01, -0.18,
        "Figura 3: Función de transferencia ADC→Lux obtenida por mínimos cuadrados; "
        "define el algoritmo de conversión implementado en el firmware.",
        transform=ax3.transAxes, fontsize=7, style='italic')

    # --- Gráfica 4: Histograma de distribución del ruido ---
    ax4 = fig.add_subplot(2, 2, 4)
    residual_lux = lux - np.polyval(np.polyfit(t, lux, 3), t)
    ax4.hist(residual_lux, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    ax4.axvline(0, color='red', linestyle='--', linewidth=1.2)
    ax4.set_xlabel("Residual de Luz [Lux]", fontsize=10)
    ax4.set_ylabel("Frecuencia", fontsize=10)
    ax4.set_title("Distribución del Ruido del Sensor LDR", fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.text(0.01, -0.18,
        "Figura 4: Histograma del ruido residual del sensor; "
        "la distribución gaussiana indica que el ruido es predominantemente térmico.",
        transform=ax4.transAxes, fontsize=7, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig("fusion_sensores.png", dpi=150, bbox_inches='tight')
    plt.show()

    # --- Cálculo de ENOB ---
    resultado_enob = calcular_enob(luz_adc)
    print("\n" + "=" * 55)
    print("  ANÁLISIS DE RESOLUCIÓN EFECTIVA")
    print("=" * 55)
    if resultado_enob:
        enob, snr_db = resultado_enob
        print(f"  SNR estimado:  {snr_db:.2f} dB")
        print(f"  ENOB estimado: {enob:.2f} bits efectivos")
        print(f"  (ADC teórico del ADS1115: 16 bits)")
        print(f"  Eficiencia de bits aprovechados: {enob/16*100:.1f}%")
    print("=" * 55)
    print(f"\n  Correlación Luz-Temperatura (Pearson r): {r:.4f}")
    print(f"  Ecuación de calibración: Lux = {coef_cal[0]:.4f} × ADC + {coef_cal[1]:.2f}")
    print("\n[OK] Gráfica guardada como 'fusion_sensores.png'")


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Práctica 4 — Fusión de Sensores")
    parser.add_argument('--puerto',   type=str, help='Puerto serial (ej. COM3 o /dev/ttyUSB0)')
    parser.add_argument('--baudrate', type=int, default=115200)
    parser.add_argument('--csv',      type=str, help='Ruta al archivo CSV con datos')
    parser.add_argument('--simular',  action='store_true', help='Usar datos simulados')
    args = parser.parse_args()

    if args.puerto:
        t, luz, temp = adquirir_serial(args.puerto, args.baudrate)
    elif args.csv:
        t, luz, temp = cargar_csv(args.csv)
    else:
        print("[INFO] Modo simulación activado (sin hardware)")
        t, luz, temp = simular_datos()

    graficar_y_analizar(t, luz, temp)
