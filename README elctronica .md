# Electrónica para Sistemas Inteligentes — Prácticas de Laboratorio

**Periodo:** 2026A  
**Repositorio:** https://github.com/leovdlz27/electronica

Scripts de análisis, simulación y visualización para las prácticas del curso.

---

## Dependencias

```bash
pip install numpy matplotlib scipy pyserial
```

| Librería | Uso |
|----------|-----|
| `numpy` | Cálculos vectoriales, regresión, estadística |
| `matplotlib` | Generación de todas las gráficas |
| `scipy` | Ajuste de curvas, filtros digitales, FFT |
| `pyserial` | Lectura serial UART desde ESP32/Arduino (solo con hardware) |

---

## Estructura del repositorio

```
electronica/
├── practica2/
│   ├── analisis_diodo.py        ← Regresión Shockley, curva I-V
│   └── README.md
├── practica3/
│   ├── simulacion_perceptron.py ← Sumador ponderado LM358, mapa de decisión
│   └── README.md
├── practica4/
│   ├── fusion_sensores.py       ← Correlación LDR + temperatura, ENOB, calibración
│   └── README.md
├── practica5/
│   ├── snr_ruido.py             ← SNR dinámico en dB, ENOB, análisis EMI
│   └── README.md
└── practica6/
    ├── alta_fidelidad.py        ← Sallen-Key, Nyquist, FFT, Min-Max Scaling
    └── README.md
```

---

## Resumen de prácticas

### P2 — Curva I-V del diodo 1N4148
**Script:** `practica2/analisis_diodo.py`  
**Librerías:** `numpy`, `matplotlib`, `scipy.optimize`

Ajuste de la ecuación de Shockley a datos experimentales del barrido 0 V–1 V. Estima la corriente de saturación `Is` y el factor de idealidad `n`. Genera gráficas en escala lineal y semilogarítmica.

```bash
cd practica2
python analisis_diodo.py
```

---

### P3 — Perceptrón analógico (LM358)
**Script:** `practica3/simulacion_perceptron.py`  
**Librerías:** `numpy`, `matplotlib`

Simula el sumador ponderado `y = -(w₁·x₁ + w₂·x₂ + b)` con saturación del amplificador. Genera mapa de calor del espacio de decisión y gráfica del barrido de bias.

```bash
cd practica3
python simulacion_perceptron.py
```

---

### P4 — Fusión de sensores y calibración
**Script:** `practica4/fusion_sensores.py`  
**Librerías:** `numpy`, `matplotlib`, `pyserial`

Adquisición de LDR y temperatura vía UART. Curva de calibración ADC→Lux, scatter plot con correlación de Pearson, histograma de ruido y cálculo de ENOB.

```bash
cd practica4
python fusion_sensores.py --simular      # sin hardware
python fusion_sensores.py --csv datos.csv
python fusion_sensores.py --puerto COM3  # con ESP32
```

---

### P5 — Ruido, SNR y ENOB
**Script:** `practica5/snr_ruido.py`  
**Librerías:** `numpy`, `matplotlib`, `scipy.signal`

Caracterización de EMI en 3 escenarios: batería de litio, fuente conmutada y fuente con blindaje capacitivo. Calcula SNR en dB y ENOB para cada caso. PSD con identificación de armónicos de 60/120 Hz.

```bash
cd practica5
python snr_ruido.py --simular
python snr_ruido.py --csv datos_ruido.csv
```

---


## Notas

- Todos los scripts tienen modo `--simular` para generar gráficas sin hardware.
- Cada ejecución genera un archivo `.png` en la misma carpeta, listo para incluir en el reporte.
- Los pies de figura en cada gráfica siguen el formato requerido: `Figura N: Descripción; Interpretación`.
