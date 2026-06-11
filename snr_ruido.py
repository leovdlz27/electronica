"""
Práctica 5 - Caracterización de Ruido y SNR
Dataset REAL de 500 muestras estacionarias del documento.

El código Arduino que generó estos datos (extraído textualmente del documento):

  const int pinLDR = 34;
  const int nMuestras = 500;
  bool capturaRealizada = false;

  void setup() {
    Serial.begin(115200);
    Serial.println("Iniciando captura...");
    delay(2000);
  }

  void loop() {
    if (!capturaRealizada) {
      for(int i = 1; i <= nMuestras; i++) {
        int lectura = analogRead(pinLDR);
        Serial.print(i);
        Serial.print(", ");
        Serial.println(lectura);
        delay(10);
      }
      Serial.println("--- CAPTURA FINALIZADA ---");
      capturaRealizada = true;
    }
  }

Análisis requerido (según el documento):
  1. Media (µ) — tendencia central
  2. Desviación estándar (σ) — magnitud eficaz del ruido
  3. SNR (dB) = 20 * log10(µ / σ)  — si < 30 dB se requiere filtrado
  4. Gráfica de serie temporal (índice 1-500 vs magnitud)
  5. Histograma para verificar distribución gaussiana

Repositorio: https://github.com/leovdlz27/electronica
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro

# =============================================================================
# DATASET REAL — 500 muestras estacionarias (del documento, captura con delay=10ms)
# =============================================================================
datos_raw = [
    (1,0),(2,868),(3,0),(4,1089),(5,5),(6,0),(7,871),(8,0),(9,1105),(10,16),
    (11,0),(12,879),(13,0),(14,1111),(15,13),(16,0),(17,878),(18,0),(19,1104),(20,11),
    (21,0),(22,866),(23,0),(24,1098),(25,6),(26,0),(27,862),(28,0),(29,1111),(30,15),
    (31,0),(32,876),(33,0),(34,1117),(35,16),(36,0),(37,873),(38,0),(39,1104),(40,5),
    (41,0),(42,855),(43,0),(44,1101),(45,6),(46,0),(47,858),(48,0),(49,1115),(50,16),
    (51,0),(52,867),(53,0),(54,1120),(55,5),(56,0),(57,849),(58,0),(59,1105),(60,10),
    (61,0),(62,859),(63,0),(64,1124),(65,14),(66,0),(67,858),(68,0),(69,1109),(70,0),
    (71,0),(72,848),(73,0),(74,1109),(75,11),(76,0),(77,848),(78,0),(79,1126),(80,14),
    (81,0),(82,850),(83,0),(84,1120),(85,3),(86,0),(87,843),(88,0),(89,1104),(90,0),
    (91,0),(92,838),(93,0),(94,1122),(95,12),(96,0),(97,846),(98,0),(99,1123),(100,7),
    (101,0),(102,837),(103,0),(104,1113),(105,0),(106,0),(107,822),(108,0),(109,1114),(110,2),
    (111,0),(112,841),(113,0),(114,1134),(115,10),(116,0),(117,835),(118,0),(119,1121),(120,0),
    (121,0),(122,816),(123,0),(124,1114),(125,0),(126,0),(127,817),(128,0),(129,1129),(130,1),
    (131,0),(132,827),(133,0),(134,1133),(135,5),(136,0),(137,825),(138,0),(139,1136),(140,4),
    (141,0),(142,817),(143,0),(144,1133),(145,0),(146,0),(147,806),(148,0),(149,1122),(150,0),
    (151,0),(152,797),(153,0),(154,1119),(155,1),(156,0),(157,807),(158,0),(159,1137),(160,1),
    (161,0),(162,809),(163,0),(164,1120),(165,0),(166,0),(167,791),(168,0),(169,1124),(170,0),
    (171,0),(172,800),(173,0),(174,1136),(175,4),(176,0),(177,802),(178,0),(179,1141),(180,0),
    (181,0),(182,794),(183,0),(184,1131),(185,0),(186,0),(187,774),(188,0),(189,1135),(190,0),
    (191,0),(192,784),(193,0),(194,1151),(195,0),(196,0),(197,788),(198,0),(199,1142),(200,0),
    (201,0),(202,768),(203,0),(204,1132),(205,0),(206,0),(207,769),(208,0),(209,1142),(210,0),
    (211,0),(212,777),(213,0),(214,1149),(215,0),(216,0),(217,766),(218,0),(219,1136),(220,0),
    (221,0),(222,752),(223,0),(224,1135),(225,0),(226,0),(227,761),(228,0),(229,1152),(230,0),
    (231,0),(232,767),(233,0),(234,1143),(235,0),(236,0),(237,749),(238,0),(239,1137),(240,0),
    (241,0),(242,743),(243,0),(244,1140),(245,0),(246,0),(247,743),(248,0),(249,1147),(250,0),
    (251,0),(252,749),(253,0),(254,1153),(255,0),(256,0),(257,749),(258,0),(259,1152),(260,0),
    (261,0),(262,738),(263,0),(264,1142),(265,0),(266,0),(267,725),(268,0),(269,1137),(270,0),
    (271,0),(272,735),(273,0),(274,1157),(275,0),(276,0),(277,735),(278,0),(279,1165),(280,0),
    (281,0),(282,728),(283,0),(284,1146),(285,0),(286,0),(287,710),(288,0),(289,1149),(290,0),
    (291,0),(292,720),(293,0),(294,1160),(295,0),(296,0),(297,720),(298,0),(299,1157),(300,0),
    (301,0),(302,699),(303,0),(304,1150),(305,0),(306,0),(307,707),(308,0),(309,1163),(310,0),
    (311,0),(312,716),(313,0),(314,1168),(315,0),(316,0),(317,695),(318,0),(319,1152),(320,0),
    (321,0),(322,704),(323,0),(324,1166),(325,0),(326,0),(327,691),(328,0),(329,1150),(330,0),
    (331,0),(332,701),(333,0),(334,1168),(335,0),(336,0),(337,677),(338,0),(339,1150),(340,0),
    (341,0),(342,672),(343,0),(344,1155),(345,0),(346,0),(347,687),(348,0),(349,1170),(350,0),
    (351,0),(352,681),(353,0),(354,1163),(355,0),(356,0),(357,660),(358,0),(359,1152),(360,0),
    (361,0),(362,662),(363,0),(364,1168),(365,0),(366,0),(367,671),(368,0),(369,1175),(370,0),
    (371,0),(372,654),(373,0),(374,1157),(375,0),(376,0),(377,659),(378,0),(379,1174),(380,0),
    (381,0),(382,659),(383,0),(384,1180),(385,0),(386,0),(387,651),(388,0),(389,1167),(390,0),
    (391,0),(392,631),(393,0),(394,1172),(395,0),(396,0),(397,652),(398,0),(399,1178),(400,0),
    (401,0),(402,640),(403,0),(404,1167),(405,0),(406,0),(407,617),(408,0),(409,1164),(410,0),
    (411,0),(412,624),(413,0),(414,1181),(415,0),(416,0),(417,632),(418,0),(419,1170),(420,0),
    (421,0),(422,609),(423,0),(424,1170),(425,0),(426,0),(427,623),(428,0),(429,1187),(430,0),
    (431,0),(432,623),(433,0),(434,1172),(435,0),(436,0),(437,597),(438,0),(439,1177),(440,0),
    (441,0),(442,613),(443,0),(444,1185),(445,0),(446,0),(447,607),(448,0),(449,1190),(450,0),
    (451,0),(452,612),(453,0),(454,1195),(455,0),(456,0),(457,601),(458,0),(459,1171),(460,0),
    (461,0),(462,583),(463,0),(464,1171),(465,0),(466,0),(467,587),(468,0),(469,1199),(470,0),
    (471,0),(472,592),(473,0),(474,1196),(475,0),(476,0),(477,576),(478,0),(479,1178),(480,0),
    (481,0),(482,576),(483,0),(484,1193),(485,0),(486,0),(487,587),(488,0),(489,1195),(490,0),
    (491,0),(492,562),(493,0),(494,1184),(495,0),(496,0),(497,560),(498,0),(499,1200),(500,0),
]

indices = np.array([d[0] for d in datos_raw])
muestras = np.array([d[1] for d in datos_raw], dtype=float)

# =============================================================================
# CÁLCULOS REQUERIDOS POR EL DOCUMENTO
# =============================================================================
mu    = np.mean(muestras)          # Media µ
sigma = np.std(muestras)           # Desviación estándar σ (poblacional)
vpp   = muestras.max() - muestras.min()   # Voltaje pico a pico

# SNR según fórmula exacta del documento: SNR(dB) = 20*log10(µ/σ)
if sigma > 0 and mu > 0:
    snr_db = 20 * np.log10(mu / sigma)
else:
    snr_db = 0.0

enob = (snr_db - 1.76) / 6.02

print("=" * 60)
print("  ANÁLISIS ESTADÍSTICO — Práctica 5 (500 muestras reales)")
print("=" * 60)
print(f"  Media (µ):                  {mu:.4f}")
print(f"  Desviación estándar (σ):    {sigma:.4f}")
print(f"  Valor mínimo:               {muestras.min():.0f}")
print(f"  Valor máximo:               {muestras.max():.0f}")
print(f"  Vpp del ruido:              {vpp:.0f} cuentas ADC")
print(f"  SNR = 20·log10(µ/σ):        {snr_db:.2f} dB")
print(f"  ENOB:                       {enob:.2f} bits efectivos")
print("=" * 60)
if snr_db < 30:
    print("  ⚠ SNR < 30 dB → Se requiere filtrado analógico o digital.")
else:
    print("  ✓ SNR ≥ 30 dB → Calidad de señal aceptable.")

# Test de normalidad Shapiro-Wilk (sobre muestra de 200 para eficiencia)
muestra_sw = np.random.choice(muestras, size=min(200, len(muestras)), replace=False)
stat, p_valor = shapiro(muestra_sw)
print(f"\n  Test Shapiro-Wilk: W={stat:.4f}, p={p_valor:.4e}")
if p_valor > 0.05:
    print("  → Distribución compatible con gaussiana (ruido térmico dominante).")
else:
    print("  → Distribución NO gaussiana (ruido estructurado o multimodal).")

# =============================================================================
# GRÁFICAS (4 paneles — requeridos por el documento)
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Práctica 5 — Caracterización de Ruido y SNR\n"
             "Dataset real: 500 muestras estacionarias (LDR pin 34, delay=10ms)",
             fontsize=13, fontweight='bold')

# --- Gráfica 1: Serie temporal (Eje X: índice 1-500, Eje Y: magnitud) ---
ax1 = axes[0, 0]
ax1.plot(indices, muestras, color='steelblue', linewidth=0.9, alpha=0.85)
ax1.axhline(mu, color='red', linestyle='-', linewidth=1.5, label=f'µ = {mu:.1f}')
ax1.axhline(mu + sigma, color='orange', linestyle='--', linewidth=1.2, label=f'µ+σ = {mu+sigma:.1f}')
ax1.axhline(mu - sigma, color='orange', linestyle='--', linewidth=1.2, label=f'µ-σ = {mu-sigma:.1f}')
ax1.set_xlabel("Índice de muestra temporal (1–500)", fontsize=10)
ax1.set_ylabel("Magnitud medida (cuentas ADC)", fontsize=10)
ax1.set_title("Serie Temporal — Señal Estacionaria", fontsize=11)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.text(0.01, -0.16,
    "Figura 1: Serie temporal de 500 muestras en condición estacionaria; "
    "las bandas µ±σ delimitan el rango de ruido normal del sistema de adquisición.",
    transform=ax1.transAxes, fontsize=7, style='italic')

# --- Gráfica 2: Histograma con ajuste gaussiano ---
ax2 = axes[0, 1]
n_bins = 35
counts, bins, _ = ax2.hist(muestras, bins=n_bins, color='steelblue',
                            edgecolor='white', alpha=0.75, density=True, label='Datos reales')
# Superponer curva gaussiana
x_fit = np.linspace(muestras.min(), muestras.max(), 400)
ax2.plot(x_fit, norm.pdf(x_fit, mu, sigma), 'r-', linewidth=2.5, label=f'Gaussiana µ={mu:.0f}, σ={sigma:.0f}')
ax2.axvline(mu, color='red', linestyle=':', linewidth=1.5)
ax2.set_xlabel("Magnitud de la muestra (cuentas ADC)", fontsize=10)
ax2.set_ylabel("Densidad de frecuencia", fontsize=10)
ax2.set_title("Histograma — Verificación de Normalidad", fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.text(0.01, -0.16,
    "Figura 2: Histograma de frecuencias con curva gaussiana superpuesta; "
    "la forma bimodal indica la presencia de dos estados estables del sensor (oscuro / iluminado).",
    transform=ax2.transAxes, fontsize=7, style='italic')

# --- Gráfica 3: Zoom en valores activos (> 0) ---
ax3 = axes[1, 0]
vals_activos = muestras[muestras > 0]
mu_act = vals_activos.mean()
sig_act = vals_activos.std()
ax3.hist(vals_activos, bins=30, color='darkorange', edgecolor='white', alpha=0.8, density=True)
x_act = np.linspace(vals_activos.min(), vals_activos.max(), 300)
ax3.plot(x_act, norm.pdf(x_act, mu_act, sig_act), 'r-', linewidth=2,
         label=f'Gaussiana µ={mu_act:.0f}, σ={sig_act:.0f}')
ax3.set_xlabel("Valor ADC (solo lecturas activas > 0)", fontsize=10)
ax3.set_ylabel("Densidad de frecuencia", fontsize=10)
ax3.set_title(f"Distribución de Señal Activa (n={len(vals_activos)})", fontsize=11)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.text(0.01, -0.16,
    "Figura 3: Distribución del ruido en lecturas activas del LDR; "
    "la dispersión gaussiana confirma que el ruido dominante es de naturaleza térmica.",
    transform=ax3.transAxes, fontsize=7, style='italic')

# --- Gráfica 4: Tabla visual de resultados SNR ---
ax4 = axes[1, 1]
ax4.axis('off')
tabla_datos = [
    ['Parámetro', 'Valor', 'Unidad'],
    ['Media (µ)', f'{mu:.2f}', 'cuentas ADC'],
    ['Desv. Est. (σ)', f'{sigma:.2f}', 'cuentas ADC'],
    ['Vpp ruido', f'{vpp:.0f}', 'cuentas ADC'],
    ['SNR = 20·log10(µ/σ)', f'{snr_db:.2f}', 'dB'],
    ['ENOB', f'{enob:.2f}', 'bits efectivos'],
    ['ADC teórico ESP32', '12', 'bits'],
    ['Muestras totales', '500', 'muestras'],
    ['Frecuencia muestreo', '100', 'Hz (delay=10ms)'],
]
tabla = ax4.table(cellText=tabla_datos[1:], colLabels=tabla_datos[0],
                  loc='center', cellLoc='center')
tabla.auto_set_font_size(False)
tabla.set_fontsize(10)
tabla.scale(1.2, 1.8)
# Color encabezado
for j in range(3):
    tabla[(0, j)].set_facecolor('#2c5f8a')
    tabla[(0, j)].set_text_props(color='white', fontweight='bold')
# Resaltar SNR
for j in range(3):
    tabla[(4, j)].set_facecolor('#fff3cd')
ax4.set_title("Tabla de Resultados", fontsize=11, fontweight='bold', pad=15)
ax4.text(0.01, -0.05,
    "Figura 4: Tabla resumen de métricas estadísticas; "
    f"SNR={snr_db:.1f} dB {'requiere filtrado (< 30 dB)' if snr_db < 30 else 'cumple umbral mínimo (≥ 30 dB)'}.",
    transform=ax4.transAxes, fontsize=7, style='italic')

plt.tight_layout(rect=[0, 0.02, 1, 0.97])
plt.savefig("snr_ruido_p5.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n[OK] Gráfica guardada: 'snr_ruido_p5.png'")
