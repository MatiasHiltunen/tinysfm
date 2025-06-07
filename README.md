graph TD
    A["🖼️ Image Stream<br/>(100s of images)"] --> B["📊 Hierarchical Feature Grid<br/>(Spatial Hashing)"]
    B --> C["🔀 GPU Feature Bank<br/>(Compressed Descriptors)"]
    C --> D["⚡ Cascade Matching<br/>(Visibility Graph)"]
    D --> E["🎯 Incremental SfM<br/>(Parallel BA)"]
    E --> F["📦 Streaming Output<br/>(COLMAP Format)"]
    
    B --> G["Novel: Octree Features"]
    D --> H["Novel: GPU Graph Matching"]
    E --> I["Novel: Parallel Bundle Adjustment"]
    
    style A fill:#f9f,stroke:#333,stroke-width:4px
    style F fill:#9f9,stroke:#333,stroke-width:4px
    style G fill:#ff9,stroke:#333,stroke-width:2px
    style H fill:#ff9,stroke:#333,stroke-width:2px
    style I fill:#ff9,stroke:#333,stroke-width:2px