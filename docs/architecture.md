# Архитектура Industrial Vision

## Общая схема

```mermaid
flowchart TB
    subgraph MAIN[" "]
        direction TB
        A["Камера / видеопоток"]
        B["Video Capture"]
        C["Preprocessing"]
        D["AI Inference"]
        E["Tracking"]
        F["ROI / Analytics"]
        G["Event Engine"]
        H["Snapshot / Evidence"]
        I["Event Storage"]
        J["REST API / JSON / MQTT"]
        K["Интерфейс мониторинга"]

        A --> B --> C --> D --> E --> F --> G --> H --> I --> J --> K
    end

    subgraph MON[" "]
        direction TB
        M1["System Monitoring"]
        M2["Camera status"]
        M3["CPU / GPU"]
        M4["RAM / Disk"]
        M5["Temperature"]
        M6["FPS / Latency"]
        M7["Errors / Logs"]

        M1 --> M2 --> M3 --> M4 --> M5 --> M6 --> M7
    end

    B --> M2
    D --> M3
    G --> M6

    style MAIN fill:none,stroke:none
    style MON fill:none,stroke:none

    classDef block fill:#ffffff,stroke:#222222,color:#111111,stroke-width:1px
    class A,B,C,D,E,F,G,H,I,J,K,M1,M2,M3,M4,M5,M6,M7 block
```

## Описание

Система построена по модульному принципу.

Video Capture отвечает за получение кадров.
Preprocessing подготавливает изображения.
AI Inference запускает модели компьютерного зрения.
Tracking связывает объекты между кадрами.
ROI / Analytics проверяет условия аналитики.
Event Engine формирует события.
Event Storage сохраняет результаты.
REST API предоставляет доступ к событиям.

Monitoring / Logging обеспечивает контроль состояния
компонентов и регистрацию ошибок.

На первом этапе архитектура спроектирована.
Реализация модулей выполняется последовательно.
