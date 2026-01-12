# 📋 Template-Anleitung

> **Für Kursteilnehmer*innen:** Diese Sektion nach dem Setup deines Projekts löschen!

## So verwenden Sie dieses Template:
Dieses Template hilft dir, dein Data Science Projekt effizient zu organisieren und zu dokumentieren. Es bietet eine gängige Struktur, um deine Arbeit zu planen, durchzuführen und zu präsentieren.

### 1. Template verwenden
Templates können in GitHub über den Button **"Use this template" -> "Create a new repository"** in der oberen rechten Ecke in ein eigenes Repository überführt werden. Nutze diese Vorlage als Inspiration und passe sie an dein Projekt an! 

### 2. Projekt klonen
Danach kannst du dein neues Repository direkt über VS Code klonen. Dazu öffnest du in VS Code die Kommando-Palette (Strg+Shift+P) bzw. (Cmd+Shift+P) auf dem Mac und gibst **"Git: Clone"** ein. Wähle dann "Clone from GitHub..." und melde dich ggf. bei GitHub an. Suche nach deinem Repository und wähle einen lokalen Ordner aus, in dem das Projekt gespeichert werden soll.

### 3. Abhängigkeiten installieren
Nachdem du das Repository geklont hast, musst du die Abhängigkeiten installieren. Öffne dazu ein neues Terminal in VS Code über die Menüleiste "Terminal"->"Neues Terminal" und führe die folgenden Befehle aus:

```bash
uv sync
```

### 4. Erweiterungen hinzufügen
Für dieses Projekt empfehlen wir die Installation der folgenden VS Code Erweiterungen:
- **Python** (Microsoft) - Bietet Unterstützung für Python-Entwicklung.
- **Jupyter** (Microsoft) - Ermöglicht das Arbeiten mit Jupyter Notebooks direkt in VS Code.
- **Even Better TOML** (tamasfe) - Verbessert die Bearbeitung von TOML-Dateien.
- **Ruff** (Astral Software) - Ein schneller Linter für Python, der dir hilft, sauberen Code zu schreiben.
- **Material Icon Theme** (PKief) - Verbessert die Dateisymbole in VS Code für eine bessere Übersicht.

Dafür kannst du den Erweiterungs-Tab in VS Code öffnen (Symbol mit den vier Quadraten auf der linken Seitenleiste) und in die Suchleiste `@recommended` eingeben. Danach sollten dir die empfohlenen Erweiterungen angezeigt werden.

### Notebooks ausführen
Im Ordner `notebooks/` findest du ein Jupyter Notebook namens `01_exploration.ipynb`, das als Ausgangspunkt für deine Datenanalyse dient. Öffne das Notebook in VS Code und wähle oben rechts dein virtuelles Environment als Kernel aus. Führe die Zellen nacheinander aus. Wenn alles geklappt hat wird das Notebook einen Datensatz von Kaggle laden und im Ordner `data/` speichern.

Von hier an kannst du mit deinem Projekt starten und die Vorlagen nach belieben anpassen.

Schaue dir für weitere Informationen zum Template die Datei [docs/project.md](./docs/project.md) an.


Für dein Projekt kannst du die folgenden Abschnitte in der `README.md` Datei anpassen, um dein Projekt zu beschreiben und zu präsentieren. Lösche anschließend diese Anleitung.

---

# [DEIN PROJEKTTITEL HIER] 🚀

> Eine kurze, prägnante Beschreibung deines Data Science Projekts in 1-2 Sätzen.

## 📊 Projektübersicht

**Problemstellung:** 
<!-- Beschreibe das Problem, das du lösen möchtest -->

**Ziel:** 
<!-- Was ist das Hauptziel deines Projekts? -->

**Methoden:** 
<!-- Welche Techniken/Algorithmen verwendest du? -->



## Setup

Klone das Repository
```bash
# Repository klonen
git clone [DEIN-REPO-LINK]
cd [REPO-NAME]
```

Installiere [uv](https://uv.dev) (falls noch nicht installiert) und synchronisiere die Abhängigkeiten
```bash
# Dependencies installieren
uv sync
```

### Ausführung

Notebooks in dieser Reihenfolge ausführen:
1. notebooks/01_exploration.ipynb
<!--
2. notebooks/02_preprocessing.ipynb
3. notebooks/03_modeling.ipynb
4. notebooks/04_results.ipynb
-->


## Summary: Intelligent Mobility Ecosystem

This project demonstrates the development of an advanced framework for managing transportation services through the synergy of **Big Data**, **Artificial Intelligence (AI)**, and **Blockchain Technology**.

#### **Core Project Components:**

1. **Data & Feature Engineering:**
   * Processed and cleaned over 2.6 million data records.
   * Extracted temporal features (**Hour, Day of the Week**) to identify complex mobility patterns and peak rush hours.
   * Optimized performance through data types management and downcasting techniques.

2. **AI-Driven Decision Logic:**
   * Implemented a **Smart Incentive Model** that evaluates trip efficiency.
   * Deployed **Surge Pricing Logic** that automatically calculates dynamic reward multipliers (e.g., 1.5x) based on the extracted time features.

3. **Software Architecture (OOP & Design Patterns):**
   * Utilized **Composition over Inheritance**: The system uses a modular design where the Taxi object "has-a" AI engine, increasing maintainability and scalability.
   * Developed a robust class structure to simulate electric vehicles and their operational logic.

4. **Blockchain Security:**
   * Integrated an immutable ledger to secure driver rewards.
   * Each transaction is cryptographically encrypted and linked using **SHA-256 hashing** to ensure absolute transparency and tamper-proof records.

#### **Conclusion:**
The result is a fully integrated system capable of autonomously evaluating trips, setting real-time economic incentives, and securely managing financial data. It serves as a prototype for mobility infrastructure in future **Smart Cities**.
