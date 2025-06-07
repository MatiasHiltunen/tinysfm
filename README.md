``` mermaid
graph TD
    A["Image Folder Input"] --> B["🔍 Auto-Detection<br/>(Resolution, Format)"]
    B --> C["Dynamic Memory<br/>Allocation"]
    C --> D["GPU Pipeline"]
    D --> E["COLMAP Output"]
    
    F["CLI Arguments"] --> G["--input folder<br/>--output folder<br/>--gpu-device<br/>--quality"]
    G --> B
    
    H["Missing Features"] --> I["Real Image Loading<br/>SIFT/ORB Features<br/>RANSAC<br/>Robust BA"]
    I --> D
    
    J["Optimizations"] --> K["Multi-GPU<br/>Streaming<br/>Progressive Output"]
    K --> D
    
    style A fill:#f9f,stroke:#333,stroke-width:4px
    style E fill:#9f9,stroke:#333,stroke-width:4px
    style H fill:#ff9,stroke:#333,stroke-width:2px
    style J fill:#9ff,stroke:#333,stroke-width:2px
```
