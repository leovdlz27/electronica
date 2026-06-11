# Electrónica para Sistemas Inteligentes — Prácticas de Laboratorio

**Periodo:** 2026A  
**Repositorio:** https://github.com/leovdlz27/electronica  

Scripts de análisis, simulación y visualización para las prácticas del curso.

---

## Estructura del repositorio

```
electronica/
├── practica2/
│   ├── analisis_diodo.py       ← Regresión Shockley, curva I-V
│   └── README.md
├── practica3/
│   ├── simulacion_perceptron.py ← Sumador ponderado LM358, mapa de decisión
│   └── README.md
├── practica4/
│   ├── fusion_sensores.py      ← Correlación LDR+temperatura, ENOB, calibración
│   └── README.md
├── practica5/
│   ├── snr_ruido.py            ← SNR dinámico, ENOB, análisis EMI
│   └── README.md
└── practica6/
    ├── alta_fidelidad.py       ← Sallen-Key, Nyquist, FFT, Min-Max Scaling
    └── README.md
```

## Resumen de prácticas

| Práctica | Tema | Script principal |
|----------|------|-----------------|
| P2 | Curva I-V diodo 1N4148, regresión Shockley | `analisis_diodo.py` |
| P3 | Perceptrón analógico con LM358 | `simulacion_perceptron.py` |
| P4 | Fusión de sensores LDR + temperatura | `fusion_sensores.py` |
| P5 | SNR, ENOB y caracterización de ruido EMI | `snr_ruido.py` |
| P6 | Filtro anti-aliasing, verificación Nyquist | `alta_fidelidad.py` |

## Dependencias globales

```bash
pip install numpy matplotlib scipy pyserial
```

## Uso general

Cada script soporta los siguientes modos:
- `--simular` → sin hardware (ideal para generar gráficas del reporte)
- `--csv archivo.csv` → con datos reales exportados
- `--puerto COMx` → lectura en tiempo real por UART

Ejemplo rápido:
```bash
cd practica5
python snr_ruido.py --simular
```
