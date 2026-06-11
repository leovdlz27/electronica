"""
Práctica 4 - Fusión de Sensores y Calibración
Dataset REAL extraído del documento de prácticas (500 muestras, LDR pin 34, ESP32).

El ESP32 manda datos por UART en formato: índice, valor_ADC
El sensor LDR en pin 34 genera lecturas que alternan entre:
  - ~0       (oscuridad / sin luz)
  - ~850-1120 (iluminación ambiental)
  - ~1089-1200 (luz directa alta intensidad)

Código Arduino que generó estos datos (del documento):
  const int pinLDR = 34;
  const int nMuestras = 500;
  int lectura = analogRead(pinLDR);
  Serial.print(i); Serial.print(", "); Serial.println(lectura);
  delay(10);

Repositorio: https://github.com/leovdlz27/electronica
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# =============================================================================
# DATASET REAL - 500 muestras del documento (índice, valor_ADC)
# Capturado con: pinLDR=34, delay=10ms, nMuestras=500
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

# =============================================================================
# PROCESAR DATASET
# =============================================================================
indices = np.array([d[0] for d in datos_raw])
adc_vals = np.array([d[1] for d in datos_raw])

# Separar por rangos según lo que genera el sensor real:
#   0        → pin sin excitación / oscuridad total
#   1-100    → ruido / valores transitorios
#   >400     → sensor excitado (luz media/alta)
mascara_luz_alta  = adc_vals > 900    # picos altos ~1089-1200 (luz directa)
mascara_luz_media = (adc_vals > 400) & (adc_vals <= 900)  # ~600-900 (ambient)
mascara_oscuro    = adc_vals < 50     # ~0 (oscuridad)

print("=" * 60)
print("  ESTADÍSTICAS DEL DATASET (Práctica 4 - 500 muestras)")
print("=" * 60)
print(f"  Total de muestras:          {len(adc_vals)}")
print(f"  Muestras en oscuridad (~0): {mascara_oscuro.sum()}")
print(f"  Muestras luz media (400-900): {mascara_luz_media.sum()}")
print(f"  Muestras luz alta (>900):   {mascara_luz_alta.sum()}")
print(f"  Valor mínimo ADC:           {adc_vals.min()}")
print(f"  Valor máximo ADC:           {adc_vals.max()}")
print(f"  Media global (µ):           {adc_vals.mean():.2f}")
print(f"  Desviación estándar (σ):    {adc_vals.std():.2f}")

# Puntos de calibración (del documento)
# Condición         | Luxes ref | ADC promedio medido
CAL_LUX = np.array([0,    50,   200,  1000])
CAL_ADC = np.array([0,    868,  1089, 1200])
coef = np.polyfit(CAL_ADC, CAL_LUX, deg=1)

def adc_a_lux(adc):
    return np.polyval(coef, adc)

lux_vals = adc_a_lux(adc_vals)

print(f"\n  Ecuación de calibración:")
print(f"  Lux = {coef[0]:.4f} × ADC + ({coef[1]:.2f})")

# =============================================================================
# GRÁFICAS
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Práctica 4 — Fusión de Sensores y Calibración\nDataset real: 500 muestras LDR (pin 34, ESP32)",
             fontsize=13, fontweight='bold')

# --- Gráfica 1: Serie temporal completa ---
ax1 = axes[0, 0]
ax1.plot(indices, adc_vals, color='steelblue', linewidth=0.8, alpha=0.85)
ax1.axhline(900, color='tomato', linestyle='--', linewidth=1.2, label='Umbral luz alta (900)')
ax1.axhline(400, color='orange', linestyle='--', linewidth=1.2, label='Umbral luz media (400)')
ax1.set_xlabel("Índice de muestra (1–500)", fontsize=10)
ax1.set_ylabel("Magnitud ADC (0–4095)", fontsize=10)
ax1.set_title("Serie Temporal — Lectura LDR Completa", fontsize=11)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.text(0.01, -0.16,
    "Figura 1: Serie temporal de 500 muestras del LDR (pin 34, ESP32); "
    "el patrón cíclico refleja las transiciones entre condiciones de iluminación del ensayo.",
    transform=ax1.transAxes, fontsize=7, style='italic')

# --- Gráfica 2: Curva de calibración ADC → Lux ---
ax2 = axes[0, 1]
adc_fine = np.linspace(0, 1250, 300)
ax2.scatter(CAL_ADC, CAL_LUX, color='forestgreen', s=100, zorder=5,
            label='Puntos de calibración (medidos)')
ax2.plot(adc_fine, np.polyval(coef, adc_fine), 'g--', linewidth=2,
         label=f'Ajuste lineal: Lux={coef[0]:.3f}·ADC+{coef[1]:.1f}')
ax2.set_xlabel("Valor ADC (crudo)", fontsize=10)
ax2.set_ylabel("Iluminancia real [Lux]", fontsize=10)
ax2.set_title("Curva de Calibración LDR", fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.text(0.01, -0.16,
    "Figura 2: Función de transferencia ADC→Lux obtenida por regresión lineal; "
    "define el algoritmo de conversión implementado en el firmware del ESP32.",
    transform=ax2.transAxes, fontsize=7, style='italic')

# --- Gráfica 3: Histograma de distribución de valores ---
ax3 = axes[1, 0]
# Mostrar solo valores > 0 para ver la distribución real del sensor
vals_positivos = adc_vals[adc_vals > 0]
ax3.hist(vals_positivos, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
ax3.set_xlabel("Valor ADC (excluyendo ceros)", fontsize=10)
ax3.set_ylabel("Frecuencia", fontsize=10)
ax3.set_title("Histograma — Distribución de Lecturas del LDR", fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.text(0.01, -0.16,
    "Figura 3: Distribución de frecuencias de lecturas activas del LDR; "
    "los dos grupos (~600-900 y ~1089-1200) corresponden a las condiciones lumínicas del ensayo.",
    transform=ax3.transAxes, fontsize=7, style='italic')

# --- Gráfica 4: Scatter Plot ADC vs muestra coloreado por condición ---
ax4 = axes[1, 1]
ax4.scatter(indices[mascara_oscuro], adc_vals[mascara_oscuro],
            color='navy', s=8, alpha=0.5, label='Oscuridad (ADC~0)')
ax4.scatter(indices[mascara_luz_media], adc_vals[mascara_luz_media],
            color='orange', s=8, alpha=0.7, label='Luz media (400–900)')
ax4.scatter(indices[mascara_luz_alta], adc_vals[mascara_luz_alta],
            color='tomato', s=8, alpha=0.7, label='Luz alta (>900)')
ax4.set_xlabel("Índice de muestra", fontsize=10)
ax4.set_ylabel("Valor ADC", fontsize=10)
ax4.set_title("Scatter Plot — Clasificación por Condición Lumínica", fontsize=11)
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.text(0.01, -0.16,
    "Figura 4: Diagrama de dispersión coloreado por condición; "
    "la tendencia decreciente en los picos indica la variación temporal de la fuente de luz durante el ensayo.",
    transform=ax4.transAxes, fontsize=7, style='italic')

plt.tight_layout(rect=[0, 0.02, 1, 0.97])
plt.savefig("fusion_sensores_p4.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n[OK] Gráfica guardada: 'fusion_sensores_p4.png'")

# =============================================================================
# ANÁLISIS DE CORRELACIÓN (scatter lux vs índice de muestra como proxy temporal)
# =============================================================================
vals_activos = adc_vals[adc_vals > 0]
indices_activos = indices[adc_vals > 0]
if len(vals_activos) > 2:
    r, p = pearsonr(indices_activos, vals_activos)
    print(f"\n  Correlación Pearson (tiempo vs ADC activo): r = {r:.4f}, p = {p:.4e}")
    if abs(r) > 0.5:
        print("  → Correlación significativa: la intensidad lumínica varía con el tiempo del ensayo.")
    else:
        print("  → Sin correlación fuerte: la iluminación fue estable durante el ensayo.")

# =============================================================================
# CÁLCULO DE ENOB (Número Efectivo de Bits)
# =============================================================================
señal_rms = np.std(adc_vals[adc_vals > 0])
ruido_rms  = np.std(adc_vals[adc_vals == 0]) if (adc_vals == 0).sum() > 1 else 1.0
if ruido_rms < 0.001:
    ruido_rms = 1.0
snr_db = 20 * np.log10(señal_rms / ruido_rms)
enob   = (snr_db - 1.76) / 6.02
print(f"\n  SNR estimado:  {snr_db:.2f} dB")
print(f"  ENOB estimado: {enob:.2f} bits efectivos (ADC teórico ESP32: 12 bits)")
