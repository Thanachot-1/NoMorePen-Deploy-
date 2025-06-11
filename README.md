# NoMorePen-Deploy-
flowchart LR
  subgraph Generator
    style Generator fill:#f9f,stroke:#333,stroke-width:1px
    Z[Input z<br/>100×1×1] --> CT1[ConvT1<br/>100→256, 7×7]
    CT1 --> BN1[BatchNorm2d(256)]
    BN1 --> R1[ReLU]
    R1 --> CT2[ConvT2<br/>256→128, 4×4, s=2, p=1]
    CT2 --> BN2[BatchNorm2d(128)]
    BN2 --> R2[ReLU]
    R2 --> CT3[ConvT3<br/>128→1, 4×4, s=2, p=1]
    CT3 --> Tanh[Tanh]
    Tanh --> OutG[Output<br/>1×28×28]
  end

  subgraph Discriminator
    style Discriminator fill:#9ff,stroke:#333,stroke-width:1px
    InD[Input img<br/>1×28×28] --> C1[Conv1<br/>1→128, 4×4, s=2, p=1]
    C1 --> L1[LeakyReLU(0.2)]
    L1 --> D1[Dropout(0.3)]
    D1 --> C2[Conv2<br/>128→256, 4×4, s=2, p=1]
    C2 --> BN3[BatchNorm2d(256)]
    BN3 --> L2[LeakyReLU(0.2)]
    L2 --> D2[Dropout(0.3)]
    D2 --> C3[Conv3<br/>256→1, 7×7]
    C3 --> Sig[Sigmoid]
    Sig --> OutD[Output Score]
  end
